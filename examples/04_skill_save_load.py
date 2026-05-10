"""
04_skill_save_load.py — Save a skill to YAML, reload and run it.

The YAML file stores the model name, prompt, and config.
API keys are never written to disk — they come from env vars at load time.

Required env vars:
    ANTHROPIC_API_KEY

Required packages:
    pip install pyyaml
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models import Model
from skills import Skill

YAML_PATH = os.path.join(os.path.dirname(__file__), "validator.yaml")

# ── Define and save ───────────────────────────────────────────────────────────
skill = Skill(
    model       = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    input       = {"messages": [{"role": "user", "parts": ["Check whether this text was written by a human or a modern LLM and rate the probability from 0 to 10. The higher the score, the more likely it is that the text was written by LLM: {text}"]}]},
    name        = "validator",
    description = "Checks the AI-ship of the text.",
)

skill.save(YAML_PATH)
print(f"saved → {YAML_PATH}\n")

# ── Reload and run ────────────────────────────────────────────────────────────
loaded = Skill.load(YAML_PATH)
print(f"loaded: {loaded}\n")

text_to_validate = "The non-perturbative regime of quantum chromodynamics remains one of the most formidable frontiers in theoretical physics, where the running coupling constant αs(μ) diverges at infrared energy scales, rendering conventional Feynman diagrammatic expansions entirely intractable and necessitating lattice QCD discretizations on Euclidean spacetime manifolds or the application of Dyson–Schwinger equations truncated at some dressed-vertex ansatz. Confinement — the empirical absence of isolated color-charged quanta — is conjectured to arise from the condensation of chromomagnetic monopoles in the dual superconductor picture, or alternatively from center vortex percolation across the SU(3) gauge group's center ℤ₃, yet a rigorous analytic proof within the Yang–Mills mass gap problem, one of the seven Millennium Prize Problems, continues to elude the community despite decades of numerical evidence from Wilson loop area-law falloff and Polyakov loop susceptibility measurements near the deconfinement crossover at T_c ≈ 155 MeV. Compounding this difficulty is the interplay between QCD and the electroweak sector within the Standard Model's SU(3)_c × SU(2)_L × U(1)_Y gauge structure, spontaneously broken to SU(3)_c × U(1)_em via the Brout–Englert–Higgs mechanism, wherein the Higgs field's vacuum expectation value v ≈ 246 GeV generates fermion masses through Yukawa couplings while leaving the gauge hierarchy problem — the quadratic sensitivity of the Higgs mass parameter m_H² to ultraviolet cutoffs Λ² — unresolved in the absence of supersymmetric partners, large extra dimensions à la Randall–Sundrum warped geometries, or compositeness from some strongly-coupled ultraviolet fixed point; meanwhile, gravitational physics at the Planck scale M_Pl ~ 10¹⁹ GeV demands a non-renormalizable completion, with candidate frameworks ranging from loop quantum gravity's spin-foam amplitudes and the EPRL vertex, to the holographic AdS/CFT correspondence — Maldacena's duality between type IIB superstring theory on AdS₅ × S⁵ and 𝒩=4 super-Yang–Mills on its conformal boundary — which has thus far offered the deepest operational bridge between quantum gravity and gauge theory, even as the black hole information paradox and the Page curve's recovery through replica wormhole saddle-point contributions in the gravitational path integral remain subjects of intense and unresolved debate."

result = loaded.run(variables={"text": text_to_validate})
print(result)
