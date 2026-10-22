# Implementation Plan

- [ ] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Body Re-ID from UNKNOWN and TEMPORARY decay with high appearance_sim
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: For deterministic bugs, scope the property to the concrete failing case(s) to ensure reproducibility
  - Test 1a — UNKNOWN re-id absence: Create an `IdentityManager`, verify at t=0 (face confirmed, track bound, `target_hist` learned). Feed 10s of updates with no face verification and `appearance_sim=None` (simulating unbound track). Assert state reaches UNKNOWN and `track_id` is `None`. Then construct an `Evidence` with `body_reid_track` set to a visible track and `body_reid_sim=0.90` (above re-id threshold). Call `update()` and assert state is STILL UNKNOWN (bug: re-id evidence is ignored on unfixed code). After fix, this should transition to TEMPORARY.
  - Test 1b — TEMPORARY decay with high appearance_sim: Create an `IdentityManager`, verify at t=0, then feed 10 seconds of updates (1s steps) with `appearance_sim=0.95`, no face verification, body track visible and not entangled. Assert confidence drops below `unknown_thresh` (0.30) and state reaches UNKNOWN (bug: body only slows decay via `effective_tau()`, cannot prevent UNKNOWN). After fix, confidence should remain above `unknown_thresh` due to boost.
  - Test 1c — Unbinding severs appearance link: Create an `IdentityManager`, verify at t=0, let confidence decay to UNKNOWN (feed 10s of updates). Assert `track_id` is `None`. Then verify that constructing `Evidence` with `bound_tr = tracker.tracks.get(None)` gives `appearance_sim = None` (confirms the severed link — pipeline cannot compute body similarity for unbound track).
  - Test 1d — head_turn_long simulation shows UNKNOWN: Run `head_turn_long` simulation and assert states include `UNKNOWN` (i.e. `["CONFIRMED", "TEMPORARY", "UNKNOWN", "CONFIRMED"]`). This confirms the bug exists end-to-end on unfixed code. After fix, UNKNOWN should NOT appear.
  - Run all tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests FAIL (this is correct - it proves the bug exists)
  - Document counterexamples found to understand root cause (e.g. "IdentityManager with appearance_sim=0.95 still decays to UNKNOWN after ~12s", "body_reid_track in Evidence is ignored")
  - Mark task complete when tests are written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Existing behaviors unchanged on non-bug-condition inputs
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy inputs (cases where isBugCondition returns false)
  - Write property-based tests capturing observed behavior patterns from Preservation Requirements
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - Test 2a — Face verification preservation: Observe that face verification sets `conf=1.0` and state to CONFIRMED. Write property test: for any `verified_track` set in Evidence, after `update()`, state is CONFIRMED and conf is 1.0. Verify passes on unfixed code.
  - Test 2b — Track loss preservation: Observe that when `track_id` not in `live_tracks`, update leads to UNKNOWN with `track_id=None`. Write property test: for any state where bound track disappears from `live_tracks`, result state is UNKNOWN and `track_id` is None. Verify passes on unfixed code.
  - Test 2c — Entanglement cap preservation: Observe that when `ambiguous=True` and belief is stale, conf is capped at `ambiguity_cap` (0.50). Write property test: for any input with `ambiguous=True` and stale belief, `conf <= ambiguity_cap`. Verify passes on unfixed code.
  - Test 2d — Mismatch penalty preservation: Observe that when `track_id` in `mismatch_tracks`, conf is multiplied by `mismatch_factor` (0.55). Write property test: for any input with mismatch, `conf` is reduced by at least `mismatch_factor`. Verify passes on unfixed code.
  - Test 2e — Low similarity preservation: Observe that `appearance_sim` below `body_boost_thresh` produces same decay as `appearance_sim=None`. Write property test: for any `appearance_sim < body_boost_thresh`, confidence after update equals confidence with `appearance_sim=None` (same dt, same other fields). Verify passes on unfixed code.
  - Test 2f — No target_hist preservation: Observe that when `target_hist` is None, no body re-id occurs. Write property test: when `body_reid_track` is None and `body_reid_sim` is 0.0 (no re-id evidence), behavior is identical to original. Verify passes on unfixed code.
  - Test 2g — Non-target tracks preservation: Observe that body re-id only applies to the target identity. Write property test: for any input where `verified_track` is a different track, re-binding behavior matches original. Verify passes on unfixed code.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [ ] 3. Fix for body re-identification when face not visible

  - [ ] 3.1 Add body re-id config parameters to IdentityConfig
    - Add `body_reid_thresh: float = 0.75` — minimum body similarity to trigger re-identification from UNKNOWN
    - Add `body_reid_conf: float = 0.40` — confidence level set on body re-identification (above `unknown_thresh = 0.30`)
    - Add `body_boost_thresh: float = 0.75` — minimum body similarity to trigger confidence boost in TEMPORARY
    - Add `body_boost_gain: float = 0.15` — confidence added per update step when body similarity is high
    - _Bug_Condition: isBugCondition(input) where input.state == UNKNOWN and input.appearance_sim >= BODY_REID_THRESH, or input.state == TEMPORARY and input.appearance_sim >= BODY_BOOST_THRESH_
    - _Expected_Behavior: Config parameters enable boost and re-id steps in update()_
    - _Preservation: New parameters have defaults; existing config usage unaffected_
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ] 3.2 Add body re-id evidence fields to Evidence dataclass
    - Add `body_reid_track: Optional[int] = None` — track ID identified from body appearance when in UNKNOWN
    - Add `body_reid_sim: float = 0.0` — body similarity score for the re-identified track
    - _Bug_Condition: Pipeline needs to pass re-id evidence to IdentityManager when in UNKNOWN_
    - _Expected_Behavior: Evidence carries body re-id track and similarity for IdentityManager to act on_
    - _Preservation: New fields default to None/0.0; existing Evidence construction unaffected_
    - _Requirements: 2.1, 2.3_

  - [ ] 3.3 Add confidence boost step in IdentityManager.update()
    - Insert after step 4 (mismatch handling), before step 5 (face verification)
    - When `self.track_id is not None` and `ev.appearance_sim is not None` and `ev.appearance_sim >= cfg.body_boost_thresh`:
    - Compute boost: `boost = cfg.body_boost_gain * (ev.appearance_sim - cfg.body_boost_thresh) / max(1e-6, 1.0 - cfg.body_boost_thresh)`
    - Apply: `self.conf = min(cap, self.conf + boost)` where `cap` respects the ambiguity cap if entangled and stale
    - Set `reason = "body appearance boost"`
    - _Bug_Condition: isBugCondition(input) where input.state == TEMPORARY and input.appearance_sim >= BODY_BOOST_THRESH and confidence_is_decaying(input)_
    - _Expected_Behavior: For TEMPORARY with high appearance_sim, confidence is boosted above what pure exponential decay would produce, extending identity persistence_
    - _Preservation: Only triggers when appearance_sim >= body_boost_thresh; below threshold, behavior is unchanged (no boost). Cap still applies for entangled/stale cases._
    - _Requirements: 2.2, 2.3, 3.3, 3.5_

  - [ ] 3.4 Add body re-identification step in IdentityManager.update()
    - Insert after step 5 (face verification), before step 6 (state mapping)
    - When `self.track_id is None` and `ev.body_reid_track is not None` and `ev.body_reid_sim >= cfg.body_reid_thresh`:
    - Re-bind: `self.track_id = ev.body_reid_track`, `self.conf = cfg.body_reid_conf`
    - Set `reason = "body re-identified"`
    - This must come before step 6 state mapping so re-bound track enters TEMPORARY tier
    - _Bug_Condition: isBugCondition(input) where input.state == UNKNOWN and input.body_visible and input.appearance_sim >= BODY_REID_THRESH_
    - _Expected_Behavior: For UNKNOWN with body re-id evidence above threshold, re-bind track and set conf to body_reid_conf (0.40), transitioning to TEMPORARY_
    - _Preservation: Only triggers when track_id is None AND body_reid_track is set AND sim >= threshold. When no re-id evidence (body_reid_track=None), behavior is unchanged. Re-id sets conf to body_reid_conf, NOT 1.0 (face verification remains only path to CONFIRMED)._
    - _Requirements: 2.1, 2.3, 3.1, 3.6, 3.7_

  - [ ] 3.5 Add body re-identification search in WholePersonPipeline.process()
    - In process(), between step 4 (feed identity manager) and step 5 (learn appearance), add body re-id search
    - When `self.identity.state == IdState.UNKNOWN` and `self.target_hist is not None`:
    - Iterate all visible, non-entangled tracks in `tracker.tracks`
    - Compute `hist_similarity(self.target_hist, tr.hist)` for each
    - Select track with highest similarity above `body_reid_thresh` (from `self.cfg.identity`)
    - Set `body_reid_track` and `body_reid_sim` in the `Evidence` before calling `self.identity.update()`
    - Note: This requires restructuring process() to compute body re-id BEFORE constructing Evidence and calling update()
    - _Bug_Condition: Pipeline doesn't search for body matches when in UNKNOWN — isBugCondition where state == UNKNOWN and body_visible and has_target_hist_
    - _Expected_Behavior: When in UNKNOWN with learned target_hist, pipeline searches visible tracks for body match and passes re-id evidence to IdentityManager_
    - _Preservation: When state is not UNKNOWN or target_hist is None, no search occurs (body_reid_track stays None, body_reid_sim stays 0.0). Existing appearance_sim computation for bound track is unchanged._
    - _Requirements: 2.1, 2.3, 3.7_

  - [ ] 3.6 Pass body re-id evidence to IdentityManager in Evidence constructor
    - Include `body_reid_track` and `body_reid_sim` in the `Evidence` constructor call in process()
    - These are `None`/`0.0` when not in UNKNOWN, when `target_hist` is None, or when no match is found
    - _Bug_Condition: Evidence must carry body re-id info from pipeline to IdentityManager_
    - _Expected_Behavior: Evidence is fully populated with body re-id fields for IdentityManager.update() to act on_
    - _Preservation: When no re-id match found, fields default to None/0.0, preserving original behavior_
    - _Requirements: 2.1, 2.3_

  - [ ] 3.7 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Body Re-ID from UNKNOWN and TEMPORARY boost
    - **IMPORTANT**: Re-run the SAME tests from task 1 - do NOT write new tests
    - The tests from task 1 encode the expected behavior
    - When these tests pass, it confirms the expected behavior is satisfied
    - Run bug condition exploration tests from step 1 (tests 1a through 1d)
    - Test 1a: UNKNOWN with body_reid_track and sim above threshold now transitions to TEMPORARY
    - Test 1b: TEMPORARY with appearance_sim=0.95 now maintains confidence above unknown_thresh
    - Test 1c: Pipeline now searches for body match when in UNKNOWN (appearance link no longer severed)
    - Test 1d: head_turn_long simulation no longer shows UNKNOWN (states become `["CONFIRMED", "TEMPORARY", "CONFIRMED"]`)
    - **EXPECTED OUTCOME**: Tests PASS (confirms bug is fixed)
    - Note: Test 1d assertion needs to be updated from `["CONFIRMED", "TEMPORARY", "UNKNOWN", "CONFIRMED"]` to `["CONFIRMED", "TEMPORARY", "CONFIRMED"]`
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ] 3.8 Verify preservation tests still pass
    - **Property 2: Preservation** - Existing behaviors unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2 (tests 2a through 2g)
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after fix (no regressions)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [ ] 4. Add unit tests for body re-id and boost behavior
  - Test `IdentityManager` body re-identification: verify UNKNOWN → TEMPORARY when `body_reid_track` is set with sim above threshold
  - Test `IdentityManager` confidence boost: verify TEMPORARY confidence increases when `appearance_sim` is above `body_boost_thresh`
  - Test `IdentityManager` no boost below threshold: verify confidence boost does not occur when `appearance_sim` is below `body_boost_thresh`
  - Test `IdentityManager` no re-id below threshold: verify re-id does not occur when `body_reid_sim` is below `body_reid_thresh`
  - Test `IdentityManager` re-id does not reach CONFIRMED: verify body re-id sets confidence to `body_reid_conf` (0.40), not 1.0
  - Test `IdentityManager` re-id respects entanglement: verify re-identified track in entangled state is capped at `ambiguity_cap`
  - Test `IdentityManager` no re-id without body_reid_track: verify re-id does not occur when `body_reid_track` is None even if `body_reid_sim` is high
  - Test all existing `IdentityTests` continue to pass unchanged
  - _Requirements: 2.1, 2.2, 3.1, 3.3, 3.5, 3.6, 3.7_

- [ ] 5. Add property-based tests for body re-id and boost
  - Generate random `appearance_sim` values in [0, 1] and verify: when sim >= `body_boost_thresh`, confidence after boost > confidence without boost; when sim < threshold, no boost (same as None)
  - Generate random sequences of Evidence with varying `body_reid_track`/`body_reid_sim` and verify: re-id only occurs when sim >= threshold and track is visible
  - Generate random game states (CONFIRMED/TEMPORARY/UNKNOWN with various appearance_sim, entanglement, track visibility) and verify preservation: for non-bug inputs, fixed output == original output
  - _Requirements: 2.1, 2.2, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [ ] 6. Add integration tests for simulation scenarios
  - Test `head_turn_long` scenario end-to-end: verify states are `["CONFIRMED", "TEMPORARY", "CONFIRMED"]` (UNKNOWN should no longer appear — body re-id maintains identity through the long head turn)
  - Test `head_turn_short` scenario still works: verify states are `["CONFIRMED", "TEMPORARY", "CONFIRMED"]` (unchanged)
  - Test crossing scenarios still never confidently wrong: verify no CONFIRMED + wrong state in crossing, crossing_similar, crossing_swap
  - Test occlusion scenario still never loses identity: verify UNKNOWN does not appear
  - Test that `target_hist` learning only occurs during clean CONFIRMED views (existing behavior preserved)
  - _Requirements: 1.1, 1.2, 2.1, 2.3, 3.1, 3.2, 3.3, 3.6_

- [ ] 7. Checkpoint - Ensure all tests pass
  - Run full test suite: `python -m unittest test_whole_person -v`
  - Ensure all exploration tests, preservation tests, unit tests, property-based tests, and integration tests pass
  - Verify no regressions in existing test suite
  - Ask the user if questions arise.
