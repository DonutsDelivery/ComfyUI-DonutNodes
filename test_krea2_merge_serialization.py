import unittest

import donut_krea2_merge_serialization as serialization


class FalseyInjectionList(list):
    def __bool__(self):
        return False


class FakeRoot:
    def __init__(self):
        self.diffusion_model = object()


class FakePatcher:
    def __init__(self, state_dict=None):
        self.model = FakeRoot()
        self.state_dict = dict(state_dict or {})
        self.injections = {}
        self.additional_models = {}
        self.attachments = {}
        self.added = []

    def clone(self):
        clone = FakePatcher(self.state_dict)
        clone.injections = {key: list(value) for key, value in self.injections.items()}
        clone.additional_models = {
            key: [item.clone() for item in values]
            for key, values in self.additional_models.items()
        }
        clone.attachments = dict(self.attachments)
        clone.added = list(self.added)
        return clone

    def get_additional_models_with_key(self, key):
        return self.additional_models.get(key, [])

    def remove_injections(self, key):
        self.injections.pop(key, None)

    def remove_additional_models(self, key):
        self.additional_models.pop(key, None)

    def remove_attachments(self, key):
        self.attachments.pop(key, None)

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        self.added.append((dict(patches), strength_patch, strength_model))

    def model_state_dict_for_saving(self, model, prefix):
        self.assert_prefix = prefix
        return dict(self.state_dict)


def make_merge(base_state=None, source_state=None):
    base = FakePatcher(base_state)
    source = FakePatcher(source_state)
    plan = (
        "diffusion_model.first",
        "diffusion_model.first.weight",
        0.0,
    )
    base.injections[serialization.KREA2_MERGE_INJECTION_KEY] = FalseyInjectionList([object()])
    base.additional_models[serialization.KREA2_MERGE_SOURCE_KEY] = [source]
    identity = serialization.KREA2_MERGE_PLAN_PREFIX + "abc"
    base.attachments[identity] = (plan,)
    return base, source, plan, identity


class Krea2MergeSerializationTests(unittest.TestCase):
    def test_falsey_runtime_injection_is_still_detected(self):
        base, source, plan, identity = make_merge()

        info = serialization.get_krea2_merge_bypass_info(base)

        self.assertIs(info[0], source)
        self.assertEqual(info[1], (plan,))
        self.assertEqual(info[2], (identity,))

    def test_runtime_without_plan_refuses_lossy_save(self):
        model = FakePatcher()
        model.injections[serialization.KREA2_MERGE_INJECTION_KEY] = FalseyInjectionList([object()])

        with self.assertRaisesRegex(RuntimeError, "save plan metadata is missing"):
            serialization.get_krea2_merge_bypass_info(model)

    def test_runtime_without_source_refuses_lossy_save(self):
        model, _source, _plan, _identity = make_merge()
        model.additional_models.clear()

        with self.assertRaisesRegex(RuntimeError, "exactly one retained model2 source"):
            serialization.get_krea2_merge_bypass_info(model)

    def test_clone_removes_only_merge_runtime_plumbing(self):
        model, _source, _plan, identity = make_merge()
        model.injections["other_runtime"] = [object()]
        model.additional_models["other_source"] = [FakePatcher()]
        model.attachments["other_attachment"] = 7

        converted = serialization.clone_without_krea2_merge_runtime(model)

        self.assertNotIn(serialization.KREA2_MERGE_INJECTION_KEY, converted.injections)
        self.assertNotIn(serialization.KREA2_MERGE_SOURCE_KEY, converted.additional_models)
        self.assertNotIn(identity, converted.attachments)
        self.assertIn("other_runtime", converted.injections)
        self.assertIn("other_source", converted.additional_models)
        self.assertEqual(converted.attachments["other_attachment"], 7)

    def test_composition_replaces_complete_direct_module_state(self):
        base_state = {
            "diffusion_model.first.weight": "base-weight",
            "diffusion_model.first.bias": "base-bias",
            "diffusion_model.first.weight_scale": "base-scale",
            "diffusion_model.first.child.weight": "base-child",
            "diffusion_model.other.weight": "base-other",
        }
        source_state = {
            "diffusion_model.first.weight": "source-weight",
            "diffusion_model.first.bias": "source-bias",
            "diffusion_model.first.weight_scale": "source-scale",
            "diffusion_model.first.input_scale": "source-input-scale",
            "diffusion_model.first.child.weight": "source-child",
            "diffusion_model.other.weight": "source-other",
        }
        base, source, plan, _identity = make_merge(base_state, source_state)

        composed, module_count, key_count = serialization.compose_krea2_merge_unet_state_dict(
            base,
            source,
            (plan,),
        )

        self.assertEqual(composed["diffusion_model.first.weight"], "source-weight")
        self.assertEqual(composed["diffusion_model.first.bias"], "source-bias")
        self.assertEqual(composed["diffusion_model.first.weight_scale"], "source-scale")
        self.assertEqual(composed["diffusion_model.first.input_scale"], "source-input-scale")
        # Child modules are not direct state of the swapped Linear and therefore
        # remain on the normal model1/partial-patch path.
        self.assertEqual(composed["diffusion_model.first.child.weight"], "base-child")
        self.assertEqual(composed["diffusion_model.other.weight"], "base-other")
        self.assertEqual(module_count, 1)
        self.assertEqual(key_count, 4)

    def test_later_bypass_lora_components_can_be_applied_only_to_swapped_weights(self):
        source = FakePatcher()
        adapter_swapped = object()
        adapter_other = object()
        components = {
            "diffusion_model.first.weight": [(adapter_swapped, 0.75)],
            "diffusion_model.other.weight": [(adapter_other, 1.0)],
        }

        converted = serialization.clone_with_regular_components(
            source,
            components,
            allowed_keys={"diffusion_model.first.weight"},
        )

        self.assertEqual(len(converted.added), 1)
        patches, strength_patch, strength_model = converted.added[0]
        self.assertIs(patches["diffusion_model.first.weight"], adapter_swapped)
        self.assertEqual(strength_patch, 0.75)
        self.assertEqual(strength_model, 1.0)


if __name__ == "__main__":
    unittest.main()
