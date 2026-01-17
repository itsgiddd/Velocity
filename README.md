# Trading Brain Conflict Resolutions

This repository includes updates to address seven conflicts in the AI brain decision pipeline. The fixes focus on preventing runtime mismatches, invalid market inputs, and unsafe trade calculations.

## Resolved Conflicts (7)
1. **Empty input protection**: Rejects decisions when H1/H4/D1 data is empty or missing entirely.  
2. **Required column validation**: Blocks execution if OHLC columns are missing in any timeframe.  
3. **Time ordering mismatch**: Ensures market data is sorted by time before analysis to avoid stale context reads.  
4. **Symbol info gaps**: Stops processing when required symbol fields (point, lot sizing attributes, tick value) are missing.  
5. **Stop-loss fallback**: Adds a safety SL based on recent swing highs/lows when patterns omit SL details.  
6. **Target distance fallback**: Establishes a measured move target using recent range or 2x risk if the pattern provides none.  
7. **Minimum history guard**: Returns a wait decision when there is insufficient bar history to compute context and patterns reliably.  

## Human-Like, Multi-Timeframe Decision Flow
- **Multi-timeframe pattern scan**: The AI brain now evaluates patterns on H1, H4, and D1 before selecting the highest-confluence setup.  
- **Persona-driven reasoning**: The decision notes include a configurable voice profile (style, tone, risk appetite) to express trade rationale in a more human-aligned narrative.  

## Roadmap: Beyond a Basic Algorithm
- **Pluggable decision policy**: The core selection logic can be swapped with a learning-based policy to move beyond fixed heuristics.  
- **Model-driven cognition**: Integrate a neural model that learns from historical outcomes and adapts to your personal trading style.  

## Pattern Training (Forex)
Use `pattern_training.py` to build a dataset from historical H1/H4/D1 candles and train a TensorFlow model when available.

**Example (training workflow):**
1. Load your FX history into three DataFrames (H1, H4, D1).
2. Build samples and train:
   ```python
   from pattern_training import build_pattern_dataset, train_tensorflow_model

   samples = build_pattern_dataset(h1_df, h4_df, d1_df, horizon=24, threshold_pct=0.002)
   model, history = train_tensorflow_model(samples, epochs=30, batch_size=64)
   model.save("fx_pattern_model.keras")
   ```

## TensorFlow Policy Integration
To resolve the policy mismatch between training and live selection, use the `TensorFlowDecisionPolicy`
so the AI brain scores patterns with the trained model.

```python
from ai_brain import AIBrain, TensorFlowDecisionPolicy
from tensorflow.keras.models import load_model

model = load_model("fx_pattern_model.keras")
policy = TensorFlowDecisionPolicy(model, threshold=0.6)
brain = AIBrain(decision_policy=policy)
```

The policy uses the same feature order as `PatternFeatureBuilder` and `build_pattern_dataset`
to keep training and inference aligned.
