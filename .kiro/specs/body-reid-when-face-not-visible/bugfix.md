# Bugfix Requirements Document

## Introduction

The face recognition and tracking system identifies a person by matching their face against an enrolled database. Once a face is verified, the system binds the identity to a body track and tracks that body. When the person turns around so their face is no longer visible to the camera, the system can no longer verify the face and relies solely on exponential confidence decay to maintain identity. The system already learns a torso color histogram while the face is front-facing, but this histogram is only used to modulate the *speed* of confidence decay — it is not used to actively re-identify the person from their body alone.

The bug is that the system loses the person's identity when they turn around for too long. The confidence decays exponentially (default tau = 4 seconds) and falls below the UNKNOWN threshold (0.30) around 8 seconds without face verification, at which point the identity is unbound and the system no longer claims to know who the person is. Even if the body track remains visible and its appearance clearly matches the learned body features, the system cannot re-identify the person from body appearance alone. It can only recover when the face becomes visible again and is re-verified against the database.

The fix should enable the system to use body appearance features (learned while the face was front-facing) to re-identify the person when the face is not visible, providing at least partial recognition from body cues rather than losing identity entirely.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN a person turns around so their face is no longer visible to the camera AND the face remains hidden for a duration exceeding the confidence decay window (approximately 8 seconds with default tau=4s and unknown_thresh=0.30) THEN the system loses the person's identity entirely, transitioning to UNKNOWN state and unbinding the track

1.2 WHEN a person's identity has decayed to UNKNOWN due to extended face loss AND the body remains visible with appearance features matching the previously learned body histogram THEN the system cannot re-identify the person from body appearance alone and remains in UNKNOWN state until the face becomes visible again

1.3 WHEN the system is in TEMPORARY state (face not visible, confidence decaying) AND body appearance similarity to the learned target histogram is high THEN the system only uses this similarity to slow the rate of confidence decay rather than actively boosting or re-asserting identity confidence

### Expected Behavior (Correct)

2.1 WHEN a person's identity has decayed to UNKNOWN due to extended face loss AND the body track remains continuously visible AND the body appearance (torso histogram) matches the previously learned target body features above a re-identification threshold THEN the system SHALL re-identify the person from body appearance and return to at least TEMPORARY state with a confidence level sufficient to maintain identity binding

2.2 WHEN the system is in TEMPORARY state (face not visible, confidence decaying) AND body appearance similarity to the learned target histogram is high (above a body re-identification threshold) THEN the system SHALL boost confidence from body appearance to slow or arrest the decay, extending the duration the identity can be maintained without a visible face

2.3 WHEN a person turns around so their face is no longer visible AND the body was previously confirmed with face verification AND body appearance features were learned during the confirmed period THEN the system SHALL use those learned body features to attempt re-identification rather than relying solely on passive confidence decay

### Unchanged Behavior (Regression Prevention)

3.1 WHEN a face is visible and matches the enrolled target THEN the system SHALL CONTINUE TO verify identity via face recognition and set confidence to 1.0 (CONFIRMED state), with face verification remaining the only path to full confidence

3.2 WHEN the body track is lost entirely (no detection or face for max_coast_s seconds) THEN the system SHALL CONTINUE TO unbind the identity and transition to UNKNOWN, since body re-identification cannot operate without a visible body track

3.3 WHEN the body track is entangled with another person (path crossing) THEN the system SHALL CONTINUE TO cap confidence at the ambiguity cap and not learn body appearance from the entangled region, preventing incorrect re-identification from mixed body regions

3.4 WHEN a visible face on the bound body does not match the target THEN the system SHALL CONTINUE TO treat this as negative evidence and reduce confidence, since face verification overrides body-based identification when available

3.5 WHEN body appearance similarity to the learned target histogram is below the body re-identification threshold AND the face is not visible THEN the system SHALL CONTINUE TO let confidence decay normally without boosting from body appearance, preventing false re-identification of a different person wearing different clothes

3.6 WHEN the system is in CONFIRMED state with a visible, verified face THEN the system SHALL CONTINUE TO learn the target's body appearance (torso histogram) from clean, non-entangled views for use in future re-identification

3.7 WHEN no body appearance features were learned (e.g., target was never confirmed long enough to build a histogram) AND the face becomes not visible THEN the system SHALL CONTINUE TO rely solely on confidence decay without body re-identification, since there are no learned features to match against
