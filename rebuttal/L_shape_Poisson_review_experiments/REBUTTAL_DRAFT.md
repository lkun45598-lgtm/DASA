# Rebuttal Draft

We thank the reviewer for the careful assessment of the manuscript. The comments have
helped us distinguish the algorithmic description, the experimental protocol, and the
scope of the conclusions more clearly. We will revise the manuscript accordingly. The
numbering below follows the detailed comments currently transcribed from the reviewer
correspondence; it must be checked against the final Editorial Manager attachment before
submission because that transcription contains a numbering gap.

## Comment 1: Clarification of Stage 3 and Stage 4

**Response.**

The two stages have distinct optimization roles. In Stage 3, the backbone network and the
weighting sub-network are updated within the same training iteration using the modified
weighted loss. The backbone parameters $\theta$ are updated by gradient descent, whereas
the sub-network parameters $\phi$ are updated by gradient ascent. In schematic form,

$$
\theta_{k+1}
 = \operatorname{Adam}_{\theta}\!\left(\theta_k,
 \nabla_{\theta}\widetilde{\mathcal L}(\theta_k,\phi_k)\right),
\qquad
\phi_{k+1}
 = \operatorname{Adam}_{\phi}\!\left(\phi_k,
 -\nabla_{\phi}\widetilde{\mathcal L}(\theta_k,\phi_k)\right).
$$

Thus, Stage 3 is a simultaneous descent--ascent update: the backbone minimizes the
weighted PINN objective and the weighting sub-network increases the emphasis on the
pointwise residuals. In Stage 4, the adaptive pointwise weighting is disabled. Only the
backbone network is fine-tuned by minimizing the original unweighted PINN loss,

$$
\mathcal L_{\mathrm{unweighted}}(\theta)
 = \mathcal L_r(\theta)+\lambda_b\mathcal L_b(\theta).
$$

The weighting sub-network is not optimized in Stage 4 and does not contribute to the
fine-tuning objective.

**Revision in the manuscript.**

We will revise the training-algorithm description and the corresponding paragraph in the
method section to state explicitly the loss, the updated parameter sets, and the opposite
optimization directions in Stages 3 and 4.

### Additional experiment: sensitivity to the four stage lengths

We agree that the iteration counts of the four stages are algorithmic hyperparameters and
should be supported experimentally. We therefore added a one-factor-at-a-time sensitivity
study on the L-shaped Poisson problem. The default schedule was fixed as
$(N_{S1},N_{S2},N_{S3},N_{S4})=(5000,2000,10000,10000)$. For each stage, we varied only
its iteration count and retained the default values for the other three stages. All runs
used the same L-shaped Poisson equation, LHS collocation protocol ($N_r=10000$ and
$N_{bc}=128\times8$), six-hidden-layer width-128 gated Modified MLP, five-layer
width-10 weighting sub-network, and loss coefficients. In Stages 3 and 4, the Adam
learning rate decayed by a factor of 0.9 every 2,000 iterations. To isolate the number of
Stage 3 updates from the Stage 4 optimization budget, Stage 4 always started at
$5.904900\times10^{-4}$, the learning rate reached after the default 10,000-step Stage 3,
and then followed the same decay schedule. Network parameters and Adam moment states were
retained at the transition. Except for the iteration count of the stage under study, all
experimental factors were held fixed across the nine schedules, including the initialization
rule, training-point sampling, network architectures, loss coefficients, optimizer settings,
initial learning rates, and learning-rate decay rule. The errors below were evaluated for the
final trained model. Each schedule, including the default baseline, was independently run
ten times; the table reports the mean and sample standard deviation over these ten runs.

| Varied stage | $(N_{S1},N_{S2},N_{S3},N_{S4})$ | Relative $L_2$ error | Maximum pointwise error |
|---|---:|---:|---:|
| Stage 1 | $(2500,2000,10000,10000)$ | $6.343157\mathrm{e}{-03}\pm4.738431\mathrm{e}{-03}$ | $6.082287\mathrm{e}{-03}\pm2.172435\mathrm{e}{-03}$ |
| Default | $(5000,2000,10000,10000)$ | $5.274183\mathrm{e}{-03}\pm4.110468\mathrm{e}{-03}$ | $5.535512\mathrm{e}{-03}\pm1.640365\mathrm{e}{-03}$ |
| Stage 1 | $(10000,2000,10000,10000)$ | $5.303326\mathrm{e}{-03}\pm3.956646\mathrm{e}{-03}$ | $5.720407\mathrm{e}{-03}\pm1.491437\mathrm{e}{-03}$ |
| Stage 2 | $(5000,1000,10000,10000)$ | $5.326818\mathrm{e}{-03}\pm3.825478\mathrm{e}{-03}$ | $5.588238\mathrm{e}{-03}\pm1.225201\mathrm{e}{-03}$ |
| Stage 2 | $(5000,4000,10000,10000)$ | $5.161844\mathrm{e}{-03}\pm3.997712\mathrm{e}{-03}$ | $5.704561\mathrm{e}{-03}\pm1.498852\mathrm{e}{-03}$ |
| Stage 3 | $(5000,2000,5000,10000)$ | $5.044061\mathrm{e}{-03}\pm3.461277\mathrm{e}{-03}$ | $5.501484\mathrm{e}{-03}\pm1.418873\mathrm{e}{-03}$ |
| Stage 3 | $(5000,2000,20000,10000)$ | $4.410008\mathrm{e}{-03}\pm3.032476\mathrm{e}{-03}$ | $5.342368\mathrm{e}{-03}\pm1.230757\mathrm{e}{-03}$ |
| Stage 4 | $(5000,2000,10000,5000)$ | $5.856680\mathrm{e}{-03}\pm4.419372\mathrm{e}{-03}$ | $5.897598\mathrm{e}{-03}\pm1.804688\mathrm{e}{-03}$ |
| Stage 4 | $(5000,2000,10000,20000)$ | $4.772101\mathrm{e}{-03}\pm2.952129\mathrm{e}{-03}$ | $5.392554\mathrm{e}{-03}\pm1.359936\mathrm{e}{-03}$ |

All nine schedules retain a mean relative $L_2$ error on the order of $10^{-3}$, and their
distributions overlap within the across-run variability. Increasing Stage 3 from 5,000 to
20,000 iterations reduces the mean error from $5.044061\mathrm{e}{-03}$ to
$4.410008\mathrm{e}{-03}$, whereas reducing Stage 1 or Stage 4 gives the largest mean
errors among the tested variants. The results indicate moderate robustness within the
tested ranges, but they do not support either global insensitivity or monotonic improvement
with a larger budget. We therefore describe the default schedule as a fixed, effective
training budget rather than as a globally optimal choice.

**Revision in the manuscript.**

We will add the stage-length table and the one-factor-at-a-time protocol to the experimental
section, state the default schedule and learning-rate control explicitly, and restrict the
conclusion to the tested ranges.

## Comment 2: Can Stage 3 use the same Adam optimizer?

**Response.**

Yes. The simultaneous descent--ascent update can be implemented with one Adam optimizer
containing two disjoint parameter groups, provided that the gradient of the weighting
sub-network is sign-reversed before its update. In the present implementation, we use two
Adam instances, one for $\theta$ and one for $\phi$, within the same training iteration.
This choice gives the two parameter sets independent first- and second-moment estimates and
makes the min--max structure explicit. It does not change the mathematical update described
above. Consequently, the relevant distinction is not whether one or two Adam objects are
used, but whether the backbone follows descent and the weighting sub-network follows ascent
on the same modified loss.

**Revision in the manuscript.**

We will add this clarification to the optimization-algorithm description and will state
explicitly that the reported implementation uses two Adam instances, while a two-parameter-
group Adam implementation is mathematically valid.

## Comment 3: Sensitivity to the inner-maximization learning rate

**Response.**

We agree that the step size used for the inner maximization can affect a one-step gradient
approximation. In the practical implementation, the corresponding control is the learning
rate of the weighting sub-network. We performed a three-seed, one-factor-at-a-time study on
the L-shaped Poisson problem. With the backbone learning rate fixed at $10^{-3}$, the
sub-network learning rate was set to $5\times10^{-4}$, $10^{-3}$, and $2\times10^{-3}$.
The resulting mean relative $L_2$ errors were, respectively,

$$
3.6611\times10^{-2},\qquad
3.7848\times10^{-2},\qquad
3.7273\times10^{-2}.
$$

Across the complete bounded sensitivity set, which also varies the sub-network architecture
and update frequency, the mean relative $L_2$ error ranges from
$3.5310\times10^{-2}$ to $4.2600\times10^{-2}$; the tested-range maximum-to-minimum
ratio is $1.2065$. These results indicate moderate robustness over the tested interval.
They do not establish global insensitivity to the inner learning rate or to arbitrary
hyperparameter choices.

**Revision in the manuscript.**

We will report the tested learning-rate interval, the number of seeds, and the bounded scope
of this sensitivity study. We will not describe the result as a comprehensive search over
the entire hyperparameter space.

## Comment 4: Normalization in Equation (8)

**Response.**

Equation (8) imposes a unit empirical mean on the pointwise weights. For a batch containing
$N$ residual points, the scaled weights are

$$
\alpha_i=\frac{o_i-o_{\min}}{o_{\max}-o_{\min}},
\qquad
\widehat\lambda_i
 =\frac{N\alpha_i}{\sum_{j=1}^{N}\alpha_j},
$$

and therefore

$$
\frac{1}{N}\sum_{i=1}^{N}\widehat\lambda_i=1.
$$

Here, the expectation in the manuscript refers to the empirical average over the current
collocation batch. The unit-mean constraint fixes the overall scale of the residual loss and
prevents the sub-network from changing the loss magnitude simply by multiplying every
weight by the same constant. A unit-variance constraint is neither required by Equation (8)
nor equivalent to it. Replacing the unit-mean normalization with a standard-deviation
normalization would define a different weighting scheme and, without an additional
positivity transformation, could yield negative weights.

**Revision in the manuscript.**

We will clarify that Equation (8) enforces unit empirical mean, not unit variance, and will
define the averaging operation explicitly.

## Comment 6: SA-PINN results in Table 1

**Response.**

We rechecked the PINN--SA-PINN comparison using the aligned PDE coefficients, sampling
protocol, network depth and width, optimization budget, seed protocol, and evaluation
metrics used for the revised comparison. The results show a problem-dependent crossover,
rather than universal superiority of either method. SA-PINN has a lower relative $L_2$
error in 7 of the 13 equations; the classical PINN has a lower error in the remaining 6.

SA-PINN is more accurate for 1D Helmholtz, both 2D Poisson cases, lid-driven cavity,
Allen--Cahn, advection--diffusion, and L-shaped Poisson. Classical PINN is more accurate
for the 1D boundary-layer Poisson problem, 1D steep Poisson, 2D Helmholtz, Burgers,
diffusion--reaction, and discontinuous advection.

This behavior is compatible with the mechanisms of pointwise self-adaptive weighting. In
the 1D boundary-layer Poisson problem, the boundary layer has a small characteristic scale;
in the 1D steep Poisson problem, $\tanh(80x)$ produces a narrow interior transition. For
2D Helmholtz, the solution must preserve a global oscillatory mode. Burgers and discontinuous
advection additionally contain nonlinear or moving transport fronts. In such cases, placing
substantially greater emphasis on a small set of high-residual points can improve local
residual fitting without reducing the global solution error, particularly under a finite
optimization budget. This is a mechanism-based interpretation; the final error table alone
does not establish a causal explanation.

The revised manuscript will remove any claim that SA-PINN must outperform classical PINN
on every equation. All valid seeds and all equations, including cases unfavorable to
SA-PINN, will be retained in the revised table.

**Revision in the manuscript.**

We will state the matched comparison protocol explicitly, report both error metrics for the
same methods, and replace universal-performance language with a problem-dependent
comparison.

## Comment 7: Meaning of the loss curves in Figures 2--7

**Response.**

The vertical axis represents the total training objective used in the corresponding stage;
it is not a relative solution error. During a weighted adversarial stage, the objective is

$$
\mathcal L_{\mathrm{weighted}}
 =\frac{1}{N}\sum_{i=1}^{N}
 \left|\widehat\lambda_i r_i\right|^2
 +\lambda_b\mathcal L_b.
$$

During unweighted warm-up and backbone fine-tuning, it is

$$
\mathcal L_{\mathrm{unweighted}}
 =\frac{1}{N}\sum_{i=1}^{N}|r_i|^2
 +\lambda_b\mathcal L_b.
$$

The weighted and unweighted quantities are different objectives. Consequently, a loss value
before and after a transition between these stages should not be interpreted as a direct
comparison of one unchanged scalar objective.

**Revision in the manuscript.**

We will label the vertical axes and figure captions according to the objective used in each
stage and will identify stage transitions in the loss histories.

## Comment 8: Limitations of the NTK interpretation

**Response.**

We agree that the NTK discussion is interpretive rather than a convergence proof for DASA.
Classical NTK analyses generally rely on infinitely wide networks, approximately linearized
training dynamics, and fixed loss weights. DASA uses finite-width networks, a nonlinear
weighting sub-network, and pointwise weights that change during training. The NTK argument
therefore provides a heuristic explanation of the observed optimization behavior; it does
not establish a formal convergence guarantee or a theorem of superiority for DASA.

**Revision in the manuscript.**

We will state these assumptions and limitations explicitly in Section 4.2 and remove wording
that could suggest that the NTK discussion is a rigorous proof.

## Comment 9: Additional reference

**Response.**

We will add the work identified by DOI `10.1016/j.enganabound.2024.105802` to the
Introduction and explain its relevance to adaptive physics-informed and boundary-value
numerical methods. The complete bibliographic record will be verified against the DOI
metadata before submission.

## Publisher request: Interactive data visualization

**Response.**

We will provide machine-readable source data for the loss histories, predicted solutions,
pointwise errors, and hyperparameter-sensitivity study, together with interactive
visualizations in the supplementary research-data package in accordance with the
publisher's instructions. The manuscript will retain static figures for archival
compatibility.

## Additional Comment: Computational cost

**Response.**

We conducted a controlled L-shaped Poisson cost comparison on an NVIDIA A100-SXM4-80GB
GPU using three random seeds. The comparison reports trainable parameter count, the average
time per update in the main Adam stage, total Adam time, and complete training time. PINN
and DASA use the same six-hidden-layer, width-128 Modified MLP backbone with the same tanh
activation and initialization scheme. The shared backbone contains 83,841 trainable
parameters. DASA additionally employs a five-hidden-layer, width-10 weighting sub-network
with 480 trainable parameters, resulting in 84,321 trainable parameters in total. Therefore,
the difference between the two configurations is limited to the DASA weighting sub-network
and its associated adversarial optimization procedure.

For this table, **main Adam-stage time per step** is the mean time for one update in the main
training stage. For DASA, this is the joint adversarial update of the backbone and weighting
sub-network; for PINN, it is the corresponding unweighted backbone update. **Total Adam
time** includes all configured Adam stages, including backbone warm-up, sub-network
pretraining when applicable, the main/adversarial stage, and backbone fine-tuning.

| Method | Trainable parameters | Main Adam-stage time per step (ms) | Total Adam time (min) | Complete training time (min) |
|---|---:|---:|---:|---:|
| PINN baseline | 83,841 | $4.3266\times10^{1}\pm1.3105$ | $1.7948\times10^{1}\pm4.0481\times10^{-1}$ | $2.4006\times10^{1}\pm1.1040$ |
| DASA | 84,321 | $4.6873\times10^{1}\pm9.5260\times10^{-1}$ | $1.9361\times10^{1}\pm2.7035\times10^{-1}$ | $2.5956\times10^{1}\pm1.0811$ |

Under this configuration, DASA increases the trainable parameter count by 0.5725% and the
complete training time by 8.1239%. The main-stage time per step and total Adam time increase
by 8.3368% and 7.8727%, respectively, because DASA performs the additional sub-network
computation and adversarial update. We will replace the manuscript's broad “on par”
statement with these quantitative measurements.

## Additional Comment: Hyperparameter sensitivity

**Response.**

We conducted a three-seed, one-factor-at-a-time sensitivity study on the L-shaped Poisson
problem. The tested factors were the weighting-subnetwork architecture $3\times5$,
$5\times10$, and $7\times20$; the backbone and sub-network learning rates over the
bounded range $5\times10^{-4}$ to $2\times10^{-3}$; and the sub-network-to-backbone
update ratios $1{:}5$, $1{:}1$, and $5{:}1$. All other settings and the optimization
budget were fixed. Each variant was evaluated immediately after the adversarial stage so
that the effect of the tested DASA choices was not obscured by later optimization.

The baseline $5\times10$ sub-network with learning rates
$(10^{-3},10^{-3})$ and a $1{:}1$ update ratio gives a mean relative $L_2$ error of
$3.7848\times10^{-2}\pm6.89\times10^{-3}$. Across the tested variants, the mean relative
$L_2$ error ranges from $3.5310\times10^{-2}$ to $4.2600\times10^{-2}$, with a
tested-range maximum-to-minimum ratio of 1.2065. The results support moderate robustness
within this bounded interval, but do not imply insensitivity to arbitrary architectures,
learning rates, or update schedules.

**Revision in the manuscript.**

We will add the sensitivity ranges, the update ratios, the number of seeds, and the
limitation of the bounded study to the experimental section. Guidance to practitioners will
be stated as an empirically supported operating range rather than as a universal default.
