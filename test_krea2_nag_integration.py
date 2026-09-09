import copy
import unittest
from unittest.mock import patch

import torch
import comfy.patcher_extension
import DonutKSamplerCFGLinear as samplers
import krea2_nag_integration as nag


WRAPPER = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL


class Model:
    def __init__(self):
        self.wrappers = {WRAPPER: {key: [object()] for key in (
            'donut_krea2_edit', 'krea2_edit', 'krea2_normalized_attention_guidance',
            'krea2_edit_normalized_attention_guidance', 'other',
        )}}

    def clone(self):
        model = Model()
        model.wrappers = copy.deepcopy(self.wrappers)
        return model

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers[wrapper_type].pop(key, None)


class CaptureNAG:
    calls = []

    def patch(self, **kwargs):
        self.calls.append(kwargs)
        return (kwargs['model'],)


class NAGIntegrationTests(unittest.TestCase):
    def setUp(self):
        CaptureNAG.calls = []
        self.registry = patch.dict(nag.nodes.NODE_CLASS_MAPPINGS, {
            'Krea2NormalizedAttentionGuidance': CaptureNAG,
            'Krea2EditNormalizedAttentionGuidance': CaptureNAG,
        })
        self.registry.start()
        self.addCleanup(self.registry.stop)
        self.negative = [[torch.ones(1, 2, 4), {'pooled_output': torch.ones(1, 4)}]]
        self.target = {'samples': torch.zeros(2, 16, 8, 12)}

    def test_disabled_needs_no_dependency_and_does_not_clone(self):
        model = Model()
        with patch.dict(nag.nodes.NODE_CLASS_MAPPINGS, {}, clear=True):
            self.assertIs(nag.apply_krea2_nag(model, self.negative), model)
            with self.assertRaisesRegex(RuntimeError, 'Install/enable'):
                nag.apply_krea2_nag(model, self.negative, nag_enabled=True)

    def test_edit_replaces_competing_wrappers_on_clone_and_passes_both_references(self):
        model = Model()
        refs = [self.target, {'samples': torch.ones(1, 16, 8, 12)}]
        image_a, image_b, mask = object(), object(), object()
        explicit_negative = [[torch.full((1, 2, 4), 7.), {}]]
        result = nag.apply_krea2_nag(
            model, self.negative, nag_enabled=True, nag_negative=explicit_negative,
            source_latent=refs, source_image=image_a, source_image_b=image_b,
            vae='vae', target_latent=self.target, nag_phi=5, nag_tau=3,
            nag_alpha=.4, nag_sigma_start=10, nag_sigma_end=.1,
            nag_ref_boost=1.5, nag_ref_boost_a=.8, nag_ref_boost_mask=mask,
            nag_fit_mode='crop (legacy)',
        )
        self.assertEqual(set(result.wrappers[WRAPPER]), {'other'})
        self.assertEqual(len(model.wrappers[WRAPPER]), 5)
        call = CaptureNAG.calls[0]
        for key, value in [('source_latent', refs[0]), ('source_latent_b', refs[1]),
                           ('source_image', image_a), ('source_image_b', image_b),
                           ('target_latent', self.target), ('nag_negative', explicit_negative),
                           ('ref_boost_mask', mask)]:
            self.assertIs(call[key], value)
        self.assertEqual((call['phi'], call['tau'], call['alpha']), (5, 3, .4))
        self.assertEqual((call['sigma_start'], call['sigma_end']), (10, .1))
        self.assertEqual((call['ref_boost'], call['ref_boost_a']), (1.5, .8))
        self.assertEqual(call['fit_mode'], 'crop (legacy)')

    def test_regular_nag_keeps_unzeroed_context_while_turbo_sampler_gets_zeros(self):
        sampler = samplers.DonutSampler()
        with patch.object(sampler, 'run_simple', return_value=(self.target, 'info')) as run:
            sampler.sample(Model(), 1, 8, 8, 4, 2, 4, 'euler', 'simple',
                           self.negative, self.negative, self.target, 1.,
                           nag_enabled=True, turbo_mode=True)
        call = CaptureNAG.calls[0]
        self.assertNotIn('source_latent', call)
        self.assertIs(call['nag_negative'], self.negative)
        self.assertEqual(run.call_args.args[3:6], (1., 1., 1.))
        sampler_neg = run.call_args.args[10]
        self.assertEqual(torch.count_nonzero(sampler_neg[0][0]), 0)
        self.assertEqual(torch.count_nonzero(sampler_neg[0][1]['pooled_output']), 0)
        self.assertTrue(torch.all(self.negative[0][0] == 1))
        self.assertTrue(torch.all(self.negative[0][1]['pooled_output'] == 1))

    def test_edit_negative_used_by_all_phases_before_zeroing(self):
        sampler = samplers.DonutSampler()
        image = torch.zeros(1, 64, 96, 3)
        refs = [self.target, self.target]
        with patch.object(samplers, 'prepare_krea2_edit', return_value=(
            Model(), self.negative, self.negative, refs, image,
        )), patch.object(samplers, 'patch_krea2_edit_model', side_effect=lambda m, *a, **kw: m), \
                patch.object(sampler, 'run_multi_model', return_value=(self.target, 'info')) as run:
            sampler.sample(Model(), 1, 8, 8, 4, 2, 4, 'euler', 'simple',
                           self.negative, 'unused neg', self.target, 1., mode='multi_model',
                           model_2=Model(), model_3=Model(), edit_mode=True,
                           source_image=image, source_image_b=image, vae='vae', clip='clip',
                           nag_enabled=True, turbo_mode=True)
        self.assertEqual(len(CaptureNAG.calls), 3)
        for call in CaptureNAG.calls:
            self.assertIs(call['nag_negative'], self.negative)
            self.assertIs(call['source_latent_b'], refs[1])
            self.assertIs(call['source_image_b'], image)
        self.assertEqual(torch.count_nonzero(run.call_args.args[10][0][0]), 0)
        self.assertEqual(run.call_args.args[3:6], (1., 1., 1.))

    def test_turbo_zeroing_is_independent_of_nag(self):
        self.assertIs(nag.sampler_negative(self.negative, False), self.negative)
        self.assertEqual(torch.count_nonzero(nag.sampler_negative(self.negative, True)[0][0]), 0)


if __name__ == '__main__':
    unittest.main()
