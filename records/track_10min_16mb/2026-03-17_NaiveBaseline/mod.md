You are modifying an existing baseline transformer training codebase for the OpenAI Parameter Golf challenge.

Your job is to implement a new MLP parameterization in the baseline model:

GOAL
Replace the dense MLP weight matrices with a low-rank "LoRA-style core + residual" parameterization, but ONLY inside the MLP layers.
Do not change tokenizer, dataset loading, training loop semantics, logging format, or evaluation behavior unless absolutely required for this feature.

HIGH-LEVEL DESIGN
For each MLP weight matrix W in every transformer block, replace the dense matrix with:

    W_hat = U @ V + R

Where:
- U and V are trainable low-rank factors
- rank r must be configurable from env/config
- R is a trainable residual term
- Start with r=32 as the default for the first run
- We want to be able to sweep r over {8, 16, 32, 64}

SCOPE
Only modify the MLP layers.
Leave attention, embeddings, output head, optimizer behavior, batch sizing, tokenizer, dataset, validation logic, and logging untouched unless needed to support the new MLP representation.

IMPORTANT INTERPRETATION
This is NOT standard LoRA added on top of a frozen dense matrix.
The dense MLP matrix should be replaced by a factorized core plus residual.
In other words, the trainable MLP effective weight is:

    W_eff = U @ V + R

There should NOT be a separate stored full dense base weight used in forward.

BASELINE ASSUMPTIONS
The baseline model is a GPT-like transformer with:
- transformer blocks
- an MLP submodule with up and down projections
- model dim and MLP dim already defined by config
- likely a Linear layer for W_up and W_down today

REQUIRED IMPLEMENTATION
1. Introduce a new module for factorized linear layers, something like:
   - FactorizedResidualLinear
   or a similarly clear name

2. That module should support:
   - in_features
   - out_features
   - rank
   - residual mode / size control
   - bias behavior matching the baseline linear layer

3. Effective forward must be equivalent to:
   - y = x @ (U @ V + R)^T (+ bias if baseline used one)
   or implemented more efficiently as:
   - core = (x @ V^T) @ U^T
   - residual = x @ R^T
   - y = core + residual (+ bias)

4. The rank r must be configurable from env/config.
   Add a config variable such as:
   - MLP_LOW_RANK_R
   default = 32

5. Add a switch to enable or disable this feature cleanly, such as:
   - USE_FACTOR_MLP=1
   so the baseline can still run unchanged if disabled.

RESIDUAL DESIGN
We need a residual term, but be thoughtful:
- Do NOT just recreate a full dense matrix and call it a residual, because that defeats compression.
- Implement the residual in a size-aware way.

For the first version, implement ONE of these two residual options, choosing the cleanest and least invasive:

Option A (preferred if simple):
- A trainable sparse-ish or masked residual with configurable small capacity

Option B (acceptable first version):
- A small trainable residual in low precision-compatible dense form, but with configurable reduced capacity

If needed for first implementation simplicity, you may implement R as a full trainable dense residual behind a config flag ONLY FOR CORRECTNESS TESTING, but clearly isolate this in code and add a TODO that this is not the final competition-efficient residual path.

However, the intended architecture is:
- low-rank core does most of the work
- residual is smaller / cheaper than a full dense matrix

If you think a practical first code path is:
- stage 1: full residual for correctness
- stage 2: compressed residual
then implement stage 1 cleanly but structure the module so stage 2 is easy.

INITIALIZATION
Be careful with initialization so training stays stable.

Suggested initialization behavior:
- initialize U and V with small random values or in a way that keeps scale similar to the original linear layer
- initialize residual R to zeros or near-zero so the model starts mostly from the low-rank path unless there is a strong reason otherwise
- preserve baseline behavior as much as possible

If there is a more stable initialization that makes W_eff initially resemble a standard initialized dense linear layer, do that and explain it in comments.

MLP INTEGRATION
Modify only the MLP projections:
- W_up
- W_down
(and gate projection too, if this baseline uses gated MLPs)

Replace their nn.Linear layers with the new factorized+residual module.

Do not change the rest of the block architecture.

CONFIG / ENV WIRING
Add env/config support for:
- USE_FACTOR_MLP=0/1
- MLP_LOW_RANK_R=32
- MLP_RESIDUAL_MODE=...
- any other minimal knobs needed

Keep names short, clear, and consistent with existing style.

LOGGING / DEBUGGING
Add lightweight startup logging that prints:
- whether factorized MLP is enabled
- chosen rank r
- residual mode
- parameter counts for one transformed MLP layer and ideally total transformed params

Do not spam logs during training.

PARAMETER COUNTING
Add a small helper or startup print that compares:
- baseline MLP parameter count
- factorized+residual MLP parameter count

This is important for sanity checking.

CORRECTNESS
Ensure shape correctness for:
- up projection
- down projection
- forward pass under distributed training

Do not break:
- torchrun
- mixed precision if baseline uses it
- checkpoint saving/loading for this run

STYLE
- Keep code minimal and readable
- Follow existing repository conventions
- Make the change easy to diff and review
- Prefer small, contained edits over large refactors

DELIVERABLES
Please produce:
1. A concise explanation of the design
2. The exact code changes
3. Any new config/env variables
4. Notes on tradeoffs / TODOs, especially around making the residual truly compression-friendly later

VERY IMPORTANT
Do not redesign the whole training stack.
Do not touch tokenizer, dataset, validation, optimizer, logging format, or attention unless required.
This is a surgical modification to the baseline MLP implementation only.

Also, before writing final code, briefly state:
- where in the baseline you will modify the MLP
- what residual implementation you chose for v1
- what assumptions you are making about the existing code structure