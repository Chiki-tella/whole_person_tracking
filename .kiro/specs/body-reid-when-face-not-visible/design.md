# Body Re-ID When Face Not Visible — Bugfix Design

## Overview

When a person turns around and their face is hidden beyond ~8 seconds, the system loses identity entirely (UNKNOWN state). The system already learns a torso color histogram (`target_hist`) during CONFIRMED state, but only uses it to modulate the *speed* of exponential confidence decay via `appearance_gain` in `effective_tau()`. It never uses body appearance to actively boost confidence or re-identify the person.

The fix adds two capabilities:
1. **TEMPORARY boost**: When in TEMPORARY state with high body similarity, actively boost confidence rather than only slowing decay — extending how long identity persists without a visible face.
2. **UNKNOWN re-identification**: When identity has decayed to UNKNOWN but the body track is still visible and matches the learned `target_hist` above a threshold, re-bind the identity to that track at TEMPORARY confidence level.

Face verification remains the only path to CONFIRMED. Body re-ID can at best restore TEMPORARY.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug — the system is in (or about to enter) UNKNOWN state despite a visible body track whose appearance matches learned features above threshold, with no learned body features being used for active re-identification
- **Property (P)**: The desired behavior — body appearance matching above threshold SHALL boost confidence (TEMPORARY) or re-identify to TEMPORARY (UNKNOWN), preventing total identity loss
- **Preservation**: All existing behaviors that must remain unchanged — face-only CONFIRMED path, entanglement caps, track loss handling, mismatch penalty, appearance learning rules
- **IdentityManager**: The class in `src/identity_state.py` that maintains the three-state identity model (CONFIRMED/TEMPORARY/UNKNOWN) with exponential confidence decay
- **WholePersonPipeline**: The class in `src/whole_person.py` that orchestrates detection, tracking, face identification, and identity management; learns `target_hist` during clean CONFIRMED views
- **target_hist**: The learned torso color histogram (`np.ndarray`) maintained by the pipeline, built during CONFIRMED state from non-entangled body views
- **appearance_sim**: The Bhattacharyya-based similarity (0–1 scale) between `target_hist` and the bound track's `torso_hist`, passed in `Evidence.appearance_sim`
- **body_reid_thresh**: New config threshold for body similarity required to trigger re-identification from UNKNOWN state
- **body_boost_thresh**: New config threshold for body similarity required to trigger confidence boost in TEMPORARY state

## Bug Details

### Bug Condition

The bug manifests when a person's face has been hidden long enough for confidence to decay below `unknown_thresh` (0.30), but the body track remains continuously visible with torso appearance matching the previously learned `target_hist`. The `IdentityManager.update()` method unbinds the track (`track_id = None`, `conf = 0.0`) and transitions to UNKNOWN. On subsequent frames, the pipeline cannot compute `appearance_sim` because `bound_tr` is `None` (the bound track was unbound), so the system has no mechanism to recover from body appearance alone.

A secondary manifestation occurs while still in TEMPORARY: even when `appearance_sim` is very high (e.g., 0.95), the system only uses it to slow decay via `effective_tau()`. It never adds confidence, so the identity still decays — just more slowly — and eventually reaches UNKNOWN.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type (state, face_visible, body_visible, appearance_sim, has_target_hist)
  OUTPUT: boolean

  -- Primary: identity lost when body could re-identify
  IF input.has_target_hist == True
     AND NOT input.face_visible
     AND input.body_visible
     AND input.appearance_sim >= BODY_REID_THRESH
     AND input.state == UNKNOWN
  THEN
    RETURN True

  -- Secondary: identity decaying when body match could boost
  IF input.has_target_hist == True
     AND NOT input.face_visible
     AND input.body_visible
     AND input.appearance_sim >= BODY_BOOST_THRESH
     AND input.state == TEMPORARY
     AND confidence_is_decaying(input)
  THEN
    RETURN True

  RETURN False
END FUNCTION
```

### Examples

- **Long head turn (8s+ face loss)**: Person A is confirmed at t=0 with a red shirt. `target_hist` is learned. At t=3, A turns around (face hidden). Confidence decays: CONFIRMED → TEMPORARY (~t=4) → UNKNOWN (~t=8). Body track remains visible with `appearance_sim ≈ 0.95`, but system stays UNKNOWN. Expected: system should re-identify from body and return to TEMPORARY.

- **Medium head turn (5s face loss, TEMPORARY)**: Person A is confirmed, then face hidden for 5 seconds. During TEMPORARY, `appearance_sim ≈ 0.90`. System only slows decay (tau increases by ~25%) but confidence still drops from 0.80 to ~0.45. Expected: system should boost confidence to slow or arrest the decay, keeping identity in TEMPORARY longer.

- **Short head turn (3s, recovers via face)**: Person A turns for 3 seconds. Confidence drops to ~0.60 (TEMPORARY). Face returns, face verification sets confidence to 1.0. This is the existing correct path and should be unaffected.

- **No learned features (edge case)**: Person was confirmed only briefly (1-2 frames). `target_hist` may be `None` or barely populated. Face is then hidden. Body re-ID cannot operate because there are no learned features to match against. System should continue with normal decay.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Face verification remains the only path to CONFIRMED state (confidence = 1.0)
- Body track loss (not in `live_tracks` for `max_coast_s` seconds) still leads to UNKNOWN — body re-ID cannot operate without a visible body track
- Entanglement caps (`ambiguity_cap`) remain unchanged — confidence is still capped while entangled and stale
- Visible non-matching face on the bound body still reduces confidence (`mismatch_factor`)
- Body appearance learning (`target_hist` update) only occurs during clean, non-entangled CONFIRMED views
- When `target_hist` is `None` (never learned), body re-identification does not occur — normal decay applies
- When body similarity is below the re-id/boost threshold, confidence decays normally without boosting

**Scope:**
All inputs that do NOT involve high body appearance similarity with learned features should be completely unaffected by this fix. This includes:
- Face-visible frames (face verification path is primary)
- Low body similarity cases (different clothing, no learned features)
- Track loss scenarios (no body to match against)
- Entangled scenarios (body appearance is unreliable during crossings)
- Non-target tracks (body re-ID only applies to the target identity)

## Hypothesized Root Cause

Based on the bug description and code analysis, the root causes are:

1. **No confidence boost mechanism in IdentityManager**: The `update()` method in `src/identity_state.py` only ever *reduces* confidence (decay, ambiguity cap, mismatch) or sets it to 1.0 (face verification). There is no code path where body appearance *adds* to confidence. The `_appearance_g()` helper maps similarity to a [-1, 1] gain, but this only feeds into `effective_tau()` to modulate decay speed — it never adds to `self.conf`.

2. **Unbinding severs the appearance link**: When confidence drops below `unknown_thresh` (step 6, else branch), `IdentityManager.update()` sets `self.track_id = None` and `self.conf = 0.0`. On the next frame, the pipeline computes `bound_tr = tracker.tracks.get(None)` which is `None`, so `appearance_sim` is set to `None` in the `Evidence`. The system has no way to match the visible body against the learned `target_hist` because the binding was severed.

3. **Pipeline doesn't search for body matches when UNKNOWN**: The pipeline in `src/whole_person.py` only computes `appearance_sim` for the *currently bound* track (`bound_tr`). When in UNKNOWN with `track_id = None`, it doesn't search visible tracks for a body match against `target_hist`. There's no mechanism to re-bind from body appearance.

4. **TEMPORARY decay is one-directional**: Even with perfect body match (`appearance_sim = 1.0`), the `effective_tau()` only increases tau by `appearance_gain * 1.0 = 0.5` (50% slower decay). With `tau = 4.0`, this gives `tau_eff = 6.0`. Confidence still decays — it just takes longer. For an 8-second face loss: `exp(-8/6) ≈ 0.26`, which is below `unknown_thresh`. The body match slows but cannot prevent the eventual drop to UNKNOWN.

## Correctness Properties

Property 1: Bug Condition - Body Re-Identification from UNKNOWN

_For any_ input where the identity is in UNKNOWN state, the face is not visible, the body track is continuously visible, learned body features exist (`target_hist` is not None), and the body appearance similarity to the learned features is above the re-identification threshold (`body_reid_thresh`), the fixed system SHALL re-bind the identity to that body track and return to at least TEMPORARY state with confidence at or above `unknown_thresh`.

**Validates: Requirements 2.1, 2.3**

Property 2: Bug Condition - Confidence Boost in TEMPORARY from Body Appearance

_For any_ input where the identity is in TEMPORARY state, the face is not visible, the body track is visible, learned body features exist, and body appearance similarity is above the boost threshold (`body_boost_thresh`), the fixed system SHALL boost confidence above what pure exponential decay (even with tau modulation) would produce, extending the duration the identity can be maintained without a visible face.

**Validates: Requirements 2.2, 2.3**

Property 3: Preservation - Face Verification Remains Only Path to CONFIRMED

_For any_ input where no face verification occurs (no `verified_track` in Evidence), the fixed system SHALL NOT reach CONFIRMED state, regardless of body appearance similarity. Body re-identification can at most restore TEMPORARY.

**Validates: Requirements 3.1, 3.6**

Property 4: Preservation - Non-Body Inputs Unchanged

_For any_ input where the bug condition does NOT hold (body similarity below threshold, no learned features, face visible, track lost, or entangled), the fixed `IdentityManager.update()` SHALL produce the same state and confidence as the original function, preserving all existing decay, cap, mismatch, and track-loss behavior.

**Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.7**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct, changes are needed in two files:

**File**: `src/identity_state.py`

**Class**: `IdentityConfig`

**Specific Changes**:
1. **Add body re-identification config parameters**:
   - `body_reid_thresh: float = 0.75` — minimum body similarity to trigger re-identification from UNKNOWN
   - `body_reid_conf: float = 0.40` — confidence level set on body re-identification (above `unknown_thresh = 0.30`)
   - `body_boost_thresh: float = 0.75` — minimum body similarity to trigger confidence boost in TEMPORARY
   - `body_boost_gain: float = 0.15` — confidence added per update step when body similarity is high (scaled by how far above threshold the similarity is)

**Class**: `Evidence`

2. **Add body re-identification evidence fields**:
   - `body_reid_track: Optional[int] = None` — track ID identified from body appearance when in UNKNOWN
   - `body_reid_sim: float = 0.0` — body similarity score for the re-identified track

**Class**: `IdentityManager`

3. **Add confidence boost step in `update()`** (after step 4 mismatch handling, before step 5 face verification):
   - When `self.track_id is not None` and `ev.appearance_sim is not None` and `ev.appearance_sim >= cfg.body_boost_thresh`:
   - Compute boost: `boost = cfg.body_boost_gain * (ev.appearance_sim - cfg.body_boost_thresh) / max(1e-6, 1.0 - cfg.body_boost_thresh)`
   - Apply: `self.conf = min(cap, self.conf + boost)` where `cap` respects the ambiguity cap if entangled and stale
   - Set `reason = "body appearance boost"`

4. **Add body re-identification step in `update()`** (after step 5 face verification, before step 6 state mapping):
   - When `self.track_id is None` and `ev.body_reid_track is not None` and `ev.body_reid_sim >= cfg.body_reid_thresh`:
   - Re-bind: `self.track_id = ev.body_reid_track`, `self.conf = cfg.body_reid_conf`
   - Set `reason = "body re-identified"`
   - This must come before the step 6 state mapping so the re-bound track enters the TEMPORARY tier

5. **Preserve existing step 6 state mapping logic**: The final mapping from confidence to state remains unchanged. If the body boost or re-id set confidence above `unknown_thresh`, the state naturally becomes TEMPORARY. If not, UNKNOWN is still reached.

**File**: `src/whole_person.py`

**Class**: `WholePersonPipeline`

6. **Add body re-identification search when in UNKNOWN** (in the `process()` method, between step 4 and step 5):
   - When `self.identity.state == IdState.UNKNOWN` and `self.target_hist is not None`:
   - Iterate all visible, non-entangled tracks in `tracker.tracks`
   - Compute `hist_similarity(self.target_hist, tr.hist)` for each
   - Select the track with the highest similarity above `body_reid_thresh`
   - Set `body_reid_track` and `body_reid_sim` in the `Evidence`

7. **Pass body re-id evidence to IdentityManager**: Include `body_reid_track` and `body_reid_sim` in the `Evidence` constructor call. These are `None`/`0.0` when not in UNKNOWN or when no match is found.

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that simulate the head_turn_long scenario and assert that the system goes UNKNOWN despite the body being visible and matching. Run these tests on the UNFIXED code to observe failures and understand the root cause. Also test the IdentityManager directly with high `appearance_sim` in TEMPORARY to show confidence still decays.

**Test Cases**:
1. **Long head turn UNKNOWN test**: Run `head_turn_long` simulation and verify states include UNKNOWN (will pass on unfixed code — confirming the bug exists). This test will need to be updated after the fix to assert UNKNOWN does NOT appear.
2. **TEMPORARY decay with high appearance_sim**: Create an `IdentityManager`, verify at t=0, then feed 10 seconds of updates with `appearance_sim=0.95` and no face verification. Assert confidence drops below `unknown_thresh` (confirms body only slows decay, doesn't prevent UNKNOWN).
3. **UNKNOWN re-id absence test**: Verify at t=0, let confidence decay to UNKNOWN (feed 10s of updates), then verify `track_id` is `None` and `appearance_sim` in subsequent Evidence is `None` (confirms the unbinding severs the appearance link).
4. **Edge case — no target_hist**: Simulate UNKNOWN state with `target_hist=None` and verify body re-id does not occur (may pass on unfixed code since there's no re-id mechanism at all).

**Expected Counterexamples**:
- `head_turn_long` simulation shows UNKNOWN in state sequence despite body being continuously visible
- IdentityManager with `appearance_sim=0.95` still decays to UNKNOWN after ~12 seconds (tau_eff=6.0, exp(-12/6)≈0.14)
- After unbinding, `appearance_sim` is `None` on subsequent frames, confirming the severed link

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := fixedUpdate(input)
  ASSERT expectedBehavior(result)
END FOR
```

Where `expectedBehavior` for the UNKNOWN re-id case asserts: `result.state == TEMPORARY` and `result.conf >= unknown_thresh` and `result.track_id is not None`.

And for the TEMPORARY boost case: `result.conf > originalDecayConf(input)` (confidence is higher than what pure decay would produce).

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT originalUpdate(input) = fixedUpdate(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for face verification, track loss, entanglement, mismatch, and low-similarity cases, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Face verification preservation**: Observe that face verification sets confidence to 1.0 and state to CONFIRMED on unfixed code, then verify this continues after fix
2. **Track loss preservation**: Observe that track loss (`live_tracks` doesn't contain `track_id`) leads to UNKNOWN on unfixed code, then verify this continues after fix
3. **Entanglement cap preservation**: Observe that entangled + stale confidence is capped at `ambiguity_cap` on unfixed code, then verify this continues after fix
4. **Mismatch penalty preservation**: Observe that visible non-matching face reduces confidence on unfixed code, then verify this continues after fix
5. **Low similarity preservation**: Observe that `appearance_sim` below `body_boost_thresh` produces same decay as `appearance_sim=None` on unfixed code, then verify this continues after fix
6. **No target_hist preservation**: Observe that when `target_hist` is None, no body re-id occurs, then verify this continues after fix

### Unit Tests

- Test `IdentityManager` body re-identification: verify UNKNOWN → TEMPORARY when `body_reid_track` is set with sim above threshold
- Test `IdentityManager` confidence boost: verify TEMPORARY confidence increases when `appearance_sim` is above `body_boost_thresh`
- Test `IdentityManager` no boost below threshold: verify confidence boost does not occur when `appearance_sim` is below `body_boost_thresh`
- Test `IdentityManager` no re-id below threshold: verify re-id does not occur when `body_reid_sim` is below `body_reid_thresh`
- Test `IdentityManager` re-id does not reach CONFIRMED: verify body re-id sets confidence to `body_reid_conf`, not 1.0
- Test `IdentityManager` re-id respects entanglement: verify re-identified track in entangled state is capped
- Test existing behaviors still pass: all tests in `IdentityTests` class continue to pass

### Property-Based Tests

- Generate random `appearance_sim` values in [0, 1] and verify: when sim >= `body_boost_thresh`, confidence after boost > confidence without boost; when sim < threshold, no boost
- Generate random sequences of Evidence with varying `body_reid_track`/`body_reid_sim` and verify: re-id only occurs when sim >= threshold and track is visible
- Generate random game states (CONFIRMED/TEMPORARY/UNKNOWN with various appearance_sim, entanglement, track visibility) and verify preservation: for non-bug inputs, fixed output == original output

### Integration Tests

- Test `head_turn_long` scenario end-to-end: verify states are `["CONFIRMED", "TEMPORARY", "CONFIRMED"]` (UNKNOWN should no longer appear — body re-id maintains identity through the long head turn)
- Test `head_turn_short` scenario still works: verify states are `["CONFIRMED", "TEMPORARY", "CONFIRMED"]` (unchanged)
- Test crossing scenarios still never confidently wrong: verify no CONFIRMED + wrong state in crossing, crossing_similar, crossing_swap
- Test occlusion scenario still never loses identity: verify UNKNOWN does not appear
- Test that `target_hist` learning only occurs during clean CONFIRMED views (existing behavior preserved)
