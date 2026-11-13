# First protected Option B V2 trial

One exploratory continuation of the saved seed-1 base; test metrics were read only after validation selection. The old base and all old results remain unchanged.

| Model | Clean ACC | ASR | Target ACC |
|---|---:|---:|---:|
| Saved raw base | 92.73% | 0.5111% | 83.30% |
| Utility plus label-agreement recovery | 92.85% | 0.5222% | 83.90% |
| Selected target-protected Option B | 92.83% | 0.4778% | 83.40% |

Selection: literal mixed direction with first-order target protection, alpha 0.25. 1,950 label-consistent fitting images entered the utility stage. This data-reuse step is an additional refinement, not verified clean recovery. Ordinary utility and subtraction effects are separate rows above.

Versus the raw base, selected clean ACC changed by +0.10 percentage points and ASR by -0.0333 points. In counts, that is 10 more correct clean test predictions out of 10,000 and 3 fewer attack successes out of 9,000. These are small one-seed differences; statistical significance is not established. Subtraction alone slightly reduced ACC relative to the utility anchor while reducing ASR. Do not attribute all pipeline gains to subtraction.

Compared descriptively with the published 93.34% / 0.56% HARVEY reference, this trial has lower ASR but still LOWER accuracy. It has not met the goal of surpassing that reference on both metrics. Compare with the user's independently reproduced baseline under matching conditions before claiming superiority.

Refinement runtime: 6.24 minutes, excluding prior base training. Authoritative detailed files are in `runs/option-b-v2-protected/harvey_option_b_v2/seed1`. The source and candidate settings were frozen before this final test evaluation. The preceding internal draft's validation target-class failures motivated adding target protection and label-agreement reuse. Treat this as exploratory and confirm fixed settings with further paired seeds.

The optional reactivation audit completed after selection. It left the base copy at 92.73% / 0.511111% and measured 92.85% / 0.477778% on the re-exposed selected-model copy. Those extra updates were not retained, did not alter the selected model, and were not used to choose the final result. The authoritative selected metrics remain 92.83% / 0.477778%.
