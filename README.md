# 🦉 Knowledge Tracing for LLMs —— A Simplified and Modular Implementation

> This repository is a simplified and modular implementation of the knowledge tracing in “[Locating and Editing Factual Associations in GPT, NeurlPS, 2022.](https://rome.baulab.info)”, providing a one-step solution for model analysis.
>

📆 More features are still in development (driven by personal interest, so they may be a bit varied and unsystematic 🤣).

## 🧑‍💻 Activation Computing and Causal Effect Storage

Based on the implementation of **hook functions** (forward hook), we have defined multiple hook functions with different functionalities. By simply registering them to the corresponding layers of LLMs by name, we can automatically log intermediate states and make modifications or replacements to these intermediate hidden states.

Using the `known_1000` dataset, which contains over 1,000 factual prompts for causal tracing and can be found in `./data`.

<u>**Pipeline:**</u>

1. **Clean run:** compute and store the intermediate states during the model’s forward pass without intervention.
2. **Corrupted run:** compute and store the intermediate states during the model’s forward pass after adding noise to the subject token’s embedding.
3. **Corrupted-with-restoration run:** add noise to the subject token’s embedding, and during the model’s forward pass, restore intermediate states to those recorded from the clean run at each position. Record the final predicted probability values on the target token.

<u>**Mediator:**</u> 

1. **Hidden states:** perform a complete restoration of the hidden states computed at a specific layer.
2. **Hidden states with Attention severed:** restore the hidden states of the input to a specific layer, but keep the output of the Attention layer in the corrupted run state.
3. **Hidden states with MLP severed:** restore the hidden states of the input to a specific layer, but keep the output of the MLP layer in the corrupted run state.

## ⏳ Candidate Filtering

Since different LLMs perform differently on the sample set, filtering is necessary to select an appropriate subset of samples for causal tracing.

Only retain questions that the model can answer correctly (where the answer is the token with the highest prediction probability) to reasonably locate and trace knowledge. This is because only when the model answers correctly can it indicate that relevant knowledge may have been recalled.

## 🧑‍🎨 Causal Tracing and Causal Effect Heatmapping

Create heatmaps for the average indirect effects obtained for the three types of mediators mentioned above. 

**P.S.** The stored intermediate states and recorded prediction probabilities under different intervention methods are saved as binary files in the `./results` folder. These files are read during causal tracing and heatmap generation, facilitating reuse.

**P.P.S.** In our current implementation, the positions within the sentence include: the first subject token, the last subject token, the "further tokens" (all tokens from after the subject token to just before the last token), and the last token.

> The causal tracing heatmaps for `LLaMA-3.2-3B` and `LLaMA-3.1-8B` are shown in the figure below, consistent with the analysis results obtained in the original paper.

🦙 **LLaMA-3.1-8B:**

![llama-3.1-8b-AIE](./results/llama-3.1-8b-AIE.png)

🦙 **LLaMA-3.2-3B:**

![llama-3.2-3b-AIE](./results/llama-3.2-3b-AIE.png)

## 🗿 Environment Requirements

All dependencies are already listed in `requirements.txt`; simply run the following command in the terminal:

```bash
pip install -r requirements.txt
```

## 💫 Quick-Start Demo

We provided a quick-start demo for the llama model, with the entire configuration transmitted via a `.yaml` file, which can be found and customized in the `./hparams` folder. Since layer naming differs between models, you’ll need to locate the correct layer names when customizing. 

The layer IDs are represented by the `[LAYER_NUM]` placeholder within the configuration file, which will be automatically recognized and replaced with the corresponding layer IDs during computation.

Once the model configuration is complete, you can enter the following command in the terminal to easily perform the above knowledge tracing process.

**P.S.** Before running, don’t forget to modify the path to the config file in the `run.sh` file.

```bash
bash run.sh
```
