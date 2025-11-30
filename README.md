Based on the implementation details in the **HST (Hierarchical Spatial-Temporal)** repository (specifically the `HST-v3 ULTRA` version by `ilicilicc`), the model avoids "forgetting" primarily due to its **Complete Lattice Core** architecture, which replaces the traditional linear memory of RNNs with a structured, multi-level graph (lattice).

Here is the breakdown of how the lattice prevents forgetting:

### 1. Structured "Field" Memory vs. Linear Decay
Standard RNNs or LSTMs process data in a linear sequence, where the memory (hidden state) at step $t$ is a compressed version of step $t-1$. Over time, information "vanishes" because it is repeatedly overwritten.
The HST Lattice model, specifically through its `FullLatticeFieldAnalyzer`, treats the history as a **complete lattice field** rather than a chain. It "extracts **ALL** levels and connection patterns" simultaneously. This means past events are not just "previous steps" to be overwritten, but distinct nodes in a lattice structure that the model can inspect directly.

### 2. Path-Weighted GNN Logic
The repository utilizes what it calls **Full Path-Weighted GNN (Graph Neural Network) Logic**.
*   In a lattice graph, information flows through multiple paths.
*   The model calculates weights for these paths, allowing it to recognize and preserve important connections between distant events.
*   Because it uses a GNN approach on this lattice, the "current" state is derived by aggregating information from the *entire* relevant structure (or "Harmonic Horizon") based on connection strength, rather than just the most recent input. This allows the model to "jump" back to relevant memories without traversing a decaying linear path.

### 3. Multi-Level Hierarchical Storage
The "HST" name implies a **Hierarchical** structure. The "Complete Multi-Level Lattice Core" likely organizes information into different layers of abstraction (e.g., short-term details vs. long-term trends).
*   **Lower Lattice Levels:** Capture immediate, high-frequency changes (short-term).
*   **Higher Lattice Levels:** Capture slow-moving, global trends (long-term).
By separating these, the long-term memory is not "washed out" by the constant noise of new inputs, effectively preventing the catastrophic forgetting common in single-layer models.

### 4. The Harmonic Horizon
The implementation mentions a **Harmonic Horizon Predictor**. This suggests the model views memory in terms of "frequencies" or "harmonics" across the lattice. Just as a sine wave persists over time, "harmonic" features extracted from the lattice provide a stable, long-lasting representation of the data's underlying patterns, which naturally resists forgetting compared to transient state vectors.

**Summary:**
It does not forget because the **Lattice** turns the temporal sequence into a **spatial graph**. This allows the model to "see" its history as a structured map (which it can query via GNN logic) rather than a fading echo in a single vector.
