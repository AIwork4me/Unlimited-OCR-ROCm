"""Patcher inserts an env-gated per-sample dump into call_Edit_dist.evaluate."""

from scripts.analysis import patch_omnidocbench_dump as P  # noqa: N812

FIXTURE = """def evaluate(self, group_info=[], save_name='default'):
        samples = self.samples
        for sample in samples:
            gt = sample.get('norm_gt') or sample['gt']
        saved_samples = _as_sample_list(samples)
        with open(f'./result/{save_name}_per_page_edit.json', 'w', encoding='utf-8') as f:
            json.dump(per_img_score, f, indent=4, ensure_ascii=False)
        return samples, {'Edit_dist': {'ALL_page_avg': up_total_avg.mean()}}
"""


def test_apply_inserts_dump_block_before_per_page_write():
    out = P.apply_dump(FIXTURE)
    assert P.DUMP_SENTINEL in out
    # dump block must come BEFORE the per_page_edit.json write
    assert out.index(P.DUMP_SENTINEL) < out.index("_per_page_edit.json")
    assert "OMNIDOCBENCH_DUMP_TEXT" in out


def test_apply_is_idempotent():
    once = P.apply_dump(FIXTURE)
    twice = P.apply_dump(once)
    assert once == twice  # no double-insert


def test_revert_removes_the_block():
    patched = P.apply_dump(FIXTURE)
    assert P.revert_dump(patched) == FIXTURE


def test_patched_output_is_valid_python():
    """apply_dump must preserve the host file's indentation (regression guard:
    an earlier splice-at-anchor-text version produced IndentationError)."""
    # the fixture is a fragment; wrap it so compile() has valid surrounding scope
    src = "import json\nimport os\n\n\n" + FIXTURE
    compile(P.apply_dump(src), "<patched>", "exec")  # must not raise


def test_patched_real_scorer_body_compiles():
    """The real scorer's evaluate() body patches to syntactically valid Python."""
    real = """def evaluate(self, group_info=[], save_name='default'):
        samples = self.samples
        for sample in samples:
            img_name = sample['img_id']
            sample['image_name'] = img_name
            gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
            pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
            upper_len = max(len(pred), len(gt))
            sample['upper_len'] = upper_len
            sample['Edit_num'] = 0
        saved_samples = samples
        if not saved_samples:
            return samples, {}
        per_img_score = {}
        with open(f'./result/{save_name}_per_page_edit.json', 'w', encoding='utf-8') as f:
            json.dump(per_img_score, f, indent=4, ensure_ascii=False)
        return samples, {}
"""
    src = "import json\nimport os\n\n\n" + real
    patched = P.apply_dump(src)
    compile(patched, "<patched-real>", "exec")  # must not raise
    # the anchor line must keep its original 8-space indentation
    assert "\n        with open(f'./result/{save_name}_per_page_edit.json'" in patched
