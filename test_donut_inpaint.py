import json
from pathlib import Path
import unittest

import torch

from donut_inpaint import rasterize_mask, masked_edit_target, composite_inpaint


def mask_data(strokes=None, image="base"):
    return json.dumps({"version": 1, "image": image, "strokes": strokes if strokes is not None else [
        {"size": .3, "points": [[.5, .25], [.5, .75]], "erase": False},
    ]})


class InpaintTests(unittest.TestCase):
    def mask(self, data=None, box=(0, 0, 100, 100), size=(100, 100), feather=0):
        return rasterize_mask(data or mask_data(), "base", (100, 100), box, size, feather)

    def test_mask_follows_crop_and_resize(self):
        mask = self.mask(box=(25, 0, 75, 100), size=(100, 200))
        self.assertEqual(tuple(mask.shape), (1, 200, 100))
        self.assertEqual(mask[0, 100, 50], 1)
        self.assertEqual(mask[0, 100, 0], 0)
        self.assertEqual(mask[0, 0, 50], 0)

    def test_eraser_removes_painted_pixels(self):
        strokes = json.loads(mask_data())["strokes"]
        strokes.append({"size": .15, "erase": True, "points": [[.5, .5]]})
        mask = self.mask(mask_data(strokes))
        self.assertEqual(mask[0, 50, 50], 0)
        self.assertEqual(mask[0, 25, 50], 1)

    def test_inversion_protects_painted_area_and_feathers_only_editable_side(self):
        data = json.loads(mask_data()); data['inverted'] = True
        normal = self.mask()
        inverse = self.mask(json.dumps(data))
        self.assertTrue(torch.equal(inverse, 1 - normal))
        soft = self.mask(json.dumps(data), feather=8)
        self.assertEqual(soft[0, 50, 50], 0)
        self.assertTrue((soft <= inverse).all())
        image = torch.rand(1, 100, 100, 3)
        output = composite_inpaint(torch.ones_like(image), {'image': image, 'mask': soft})
        self.assertTrue(torch.equal(output[0, 50, 50], image[0, 50, 50]))

    def test_rectangle_drag_direction_and_eraser(self):
        forward = {'shape': 'rectangle', 'size': .08, 'points': [[.2, .3], [.8, .7]]}
        reverse = dict(forward, points=list(reversed(forward['points'])))
        expected = self.mask(mask_data([forward]))
        self.assertTrue(torch.equal(expected, self.mask(mask_data([reverse]))))
        self.assertEqual(expected[0, 50, 50], 1)
        self.assertEqual(expected[0, 10, 50], 0)
        erased = self.mask(mask_data([forward, {'size': .1, 'erase': True, 'points': [[.5, .5]]}]))
        self.assertEqual(erased[0, 50, 50], 0)
        self.assertEqual(erased[0, 40, 30], 1)

    def test_softness_is_inward_and_keeps_outside_exactly_zero(self):
        hard, soft = self.mask(), self.mask(feather=8)
        self.assertTrue(torch.equal(soft[hard == 0], hard[hard == 0]))
        self.assertTrue(((soft > 0) & (soft < 1)).any())
        self.assertTrue((soft <= hard).all())

    def test_stale_empty_erased_and_out_of_crop_masks_are_rejected(self):
        for data in [mask_data(image="different"), mask_data([]), "bad json",
                     mask_data([{"size": .2, "points": [[.5, .5]], "erase": True}])]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.mask(data)
        with self.assertRaises(ValueError):
            self.mask(box=(0, 0, 20, 100))

    def test_nonfinite_and_out_of_bounds_strokes_are_rejected(self):
        for point in [[float('nan'), .5], [2, .5]]:
            with self.assertRaises(ValueError):
                self.mask(mask_data([{"size": .2, "points": [point]}]))

    def test_target_preserves_base_latent_and_metadata_across_batch(self):
        target = {"samples": torch.zeros(3, 4, 8, 8), "batch_index": [4, 5, 6]}
        base = {"samples": torch.rand(1, 16, 8, 8)}
        identity = {"samples": torch.ones(1, 16, 8, 8)}
        mask = self.mask()
        result = masked_edit_target(target, [base, identity], {"mask": mask})
        self.assertEqual(tuple(result['samples'].shape), (3, 16, 8, 8))
        self.assertTrue(torch.equal(result['samples'][2], base['samples'][0]))
        self.assertTrue(torch.equal(result['noise_mask'][1], mask[0]))
        self.assertEqual(result['batch_index'], [4, 5, 6])
        self.assertNotIn('noise_mask', target)
        self.assertEqual(target['samples'].sum(), 0)

    def test_composite_preserves_unselected_pixels_exactly_for_batch(self):
        base = torch.rand(1, 100, 100, 3)
        mask = self.mask(feather=4)
        edited = torch.rand(2, 100, 100, 3)
        result = composite_inpaint(edited, {"image": base, "mask": mask})
        outside = (mask[0] == 0)
        for i in range(2):
            self.assertTrue(torch.equal(result[i][outside], base[0][outside]))
        self.assertTrue(torch.allclose(result[0, 50, 50], edited[0, 50, 50]))

    def test_single_frame_vae_latent_becomes_image_target_without_mutating_reference(self):
        target = {"samples": torch.zeros(2, 4, 8, 12)}
        samples = torch.rand(1, 16, 1, 8, 12)
        for source in [{"samples": samples}, [{"samples": samples}, {"samples": torch.ones_like(samples)}]]:
            with self.subTest(dual_reference=isinstance(source, list)):
                result = masked_edit_target(target, source, {"mask": self.mask()})
                self.assertEqual(tuple(result['samples'].shape), (2, 16, 8, 12))
                self.assertTrue(torch.equal(result['samples'][1], samples[0, :, 0]))
                self.assertEqual(tuple(samples.shape), (1, 16, 1, 8, 12))
                self.assertEqual(tuple(result['noise_mask'].shape), (2, 100, 100))

    def test_actual_video_and_multiple_base_images_are_rejected(self):
        for shape in [(1, 16, 2, 8, 12), (2, 16, 1, 8, 12), (2, 16, 8, 12)]:
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'received'):
                masked_edit_target({'samples': torch.zeros(1, 4, 8, 12)},
                    {'samples': torch.zeros(shape)}, {'mask': self.mask()})

    def test_upscale_and_disabled_composite(self):
        base = torch.full((1, 100, 100, 3), .25)
        result = composite_inpaint(torch.ones(1, 200, 200, 3), {"image": base, "mask": self.mask()})
        self.assertTrue(torch.equal(result[0, :10], torch.full((10, 200, 3), .25)))
        self.assertEqual(result[0, 100, 100, 0], 1)
        self.assertIs(composite_inpaint(base, None), base)

    def test_workflow_preserves_every_result_stage_and_routes_mask_to_sampler(self):
        workflow = json.loads((Path(__file__).parent / 'workflows/v5/DonutWF_v5.json').read_text())
        engine = next(g for g in workflow['definitions']['subgraphs']
                      if any(n['type'] == 'DonutSampler' for n in g['nodes']))
        nodes = {n['id']: n for n in engine['nodes']}
        links = {e['id']: e for e in engine['links']}
        inpaint_slot = next(i for i, p in enumerate(engine['inputs']) if p['name'] == 'edit_inpaint')
        for node in nodes.values():
            if node['type'] in ('VAEDecode', 'DonutTiledUpscale', 'DonutFaceDetailer', 'DonutSeedVR2Upscale'):
                outgoing = node['outputs'][0]['links']
                self.assertGreaterEqual(len(outgoing), 1)
                for outgoing_id in outgoing:
                    composite = nodes[links[outgoing_id]['target_id']]
                    if composite['type'] == 'DonutSeedVR2Upscale':
                        post_links = composite['outputs'][0]['links']
                        self.assertEqual(len(post_links), 1)
                        composite = nodes[links[post_links[0]]['target_id']]
                    self.assertEqual(composite['type'], 'DonutInpaintComposite')
            if node['type'] in ('DonutSampler', 'DonutInpaintComposite'):
                mask_input = next(p for p in node['inputs'] if p['type'] == 'DONUT_INPAINT')
                edge = links[mask_input['link']]
                self.assertEqual((edge['origin_id'], edge['origin_slot']), (-10, inpaint_slot))



class OutpaintTests(unittest.TestCase):
    def make(self, **updates):
        from PIL import Image
        from donut_inpaint import prepare_outpaint
        doc = {'version':1,'image':'base','strokes':[],
               'outpaint':{'scale':1,'x':0,'y':.5,'overlap':0}}
        doc.update(updates)
        return prepare_outpaint(Image.new('RGB',(100,200),(40,80,120)),json.dumps(doc),'base',(200,200),8)

    def test_canvas_budget_and_protected_half(self):
        out = self.make()
        self.assertEqual(tuple(out['image'].shape),(1,200,200,3))
        self.assertEqual(tuple(out['mask'].shape),(1,200,200))
        self.assertTrue(torch.all(out['mask'][:,:,:100] == 0))
        self.assertTrue(torch.all(out['mask'][:,:,100:] == 1))
        result=composite_inpaint(torch.ones_like(out['image']),out)
        self.assertTrue(torch.equal(result[:,:,:100],out['image'][:,:,:100]))
        self.assertTrue(torch.all(result[:,:,100:] == 1))

    def test_overlap_feathers_inside_a_and_never_leaks_base_into_new_space(self):
        out=self.make(outpaint={'scale':1,'x':0,'y':.5,'overlap':16})
        self.assertTrue(torch.all(out['mask'][:,:,:84] == 0))
        self.assertTrue(torch.any((out['mask'][:,:,84:100]>0)&(out['mask'][:,:,84:100]<1)))
        self.assertTrue(torch.all(out['mask'][:,:,100:] == 1))
        self.assertEqual(out['mask'][0,0,0],0) # no overlap along canvas edge

    def test_erase_protects_a_but_cannot_remove_required_new_space(self):
        out=self.make(strokes=[{'size':1,'shape':'rectangle','erase':True,'points':[[0,0],[1,1]]}],
                      outpaint={'scale':1,'x':0,'y':.5,'overlap':16})
        self.assertTrue(torch.all(out['mask'][:,:,:100] == 0))
        self.assertTrue(torch.all(out['mask'][:,:,100:] == 1))

    def test_right_placement_and_paint_over_a(self):
        out=self.make(outpaint={'scale':1,'x':1,'y':.5,'overlap':0},
                      strokes=[{'size':.1,'points':[[.8,.5]]}])
        self.assertTrue(torch.all(out['mask'][:,:,:100]==1))
        self.assertEqual(out['mask'][0,20,150],0)
        self.assertGreater(out['mask'][0,100,160],0)

    def test_invalid_placement_and_wrong_image_rejected(self):
        for key,value in [('scale',0),('scale',float('nan')),('x',-1),('y',2),('overlap',129)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                self.make(outpaint={**{'scale':1,'x':0,'y':.5,'overlap':0},key:value})
        with self.assertRaises(ValueError):self.make(image='other')

if __name__ == "__main__":
    unittest.main()
