import unittest

from src.workloads.profiles import filter_profiles, get_profile, resolve_profile_name


class ProfileRegistryTests(unittest.TestCase):
    def test_coding_agent_is_alias_for_coding_singleturn(self):
        self.assertEqual(resolve_profile_name("coding-agent"), "coding-singleturn")
        self.assertEqual(get_profile("coding-agent").name, "coding-singleturn")
        self.assertEqual(get_profile("coding-singleturn").dataset, "jsonl")

    def test_legacy_multiturn_profiles_remain_explicitly_runnable(self):
        self.assertEqual(get_profile("chat-multiturn-long").dataset, "sharegpt-multi-turn")
        self.assertEqual(get_profile("swebench-multiturn-short").dataset, "swebench-multi-turn")
        self.assertEqual(get_profile("terminalbench-multiturn-medium").dataset, "terminalbench-multi-turn")
        self.assertEqual(get_profile("osworld-multiturn-long").dataset, "osworld-multi-turn")

    def test_distributional_profiles_are_active_after_runner_wiring(self):
        active = set(filter_profiles(turn_style="multi-turn"))
        all_multi = set(filter_profiles(turn_style="multi-turn", include_inactive=True))

        canonical = {
            "chat-multiturn",
            "swebench-multiturn",
            "terminalbench-multiturn",
            "osworld-multiturn",
        }
        self.assertTrue(canonical.issubset(all_multi))
        self.assertTrue(canonical.issubset(active))


if __name__ == "__main__":
    unittest.main()
