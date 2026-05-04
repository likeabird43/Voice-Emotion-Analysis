# Voice Context-Aware Smartphone Suggestion Prototype

## Overview
This project is a lightweight AI UX prototype that infers device-relevant user state from speech cues and generates confidence-aware smartphone suggestions.

The system combines:
- a general acoustic inference pipeline based on speech features,
- optional baseline-aware personalization using a neutral user voice sample,
- and simple text-hint fusion for contextual support.

## Problem
Current voice assistants mainly respond to explicit commands.
However, users often express their needs indirectly through natural speech such as:
- "I’m tired today."
- "I’m really busy."
- "I’m in a good mood."

In realistic usage, these signals are noisy, uncertain, and highly user-dependent.

## Solution
This prototype explores a suggestion-first smartphone UX approach.

Instead of forcing a hard emotion label and directly changing device behavior, the system:
1. extracts acoustic speech features,
2. infers a device-relevant user state,
3. estimates a confidence score,
4. and generates adaptive smartphone suggestions.

## Technical Approach
The prototype uses the following acoustic features:
- pitch
- RMS energy
- spectral centroid
- speech-rate proxy

It first applies general rule-based inference and then optionally refines interpretation using a neutral baseline sample from the same user.

This design combines general inference with lightweight personalization while remaining feasible for a short prototype.

## Current Limitations
Initial tests on noisy real-world recordings showed that low-energy states such as tiredness can still overlap with stressed speech patterns, especially when speech rate remains high.

This limitation supports the choice of confidence-aware suggestion UX rather than irreversible automation.

## Future Work
Possible future improvements include:
- automatic speech-to-text integration instead of manual text hints,
- more robust baseline calibration across multiple neutral samples,
- lightweight confidence calibration,
- and testing with more users and more diverse noisy environments.