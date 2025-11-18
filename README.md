# Domain-Specific Mini-LLM for Genomic Mutation Prediction

![GitHub top language](https://img.shields.io/github/languages/top/Ankit6174/Domain-Specific-Mini-LLM-for-Genomic-Mutation-Prediction?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/Ankit6174/Domain-Specific-Mini-LLM-for-Genomic-Mutation-Prediction?style=for-the-badge)

An encoder only transformer model that predicts the significance and clinical impact of DNA mutations directly from raw sequence data.

## The Problem

One of the greatest challenges in genomics is predicting the clinical significance of DNA mutations. The amount of genetic data and variability of DNA sequence make it difficult for researchers and clinicians to rapidly assess the mutations that are likely to be pathogenic. This proposal will address this problem by employing a domain-specific transformer model to generate predictions rapidly and accurately enough to facilitate research and clinical decisions.

## Key Features

* **Data Visualization:** Displays key metrics like GC/AT content, nucleotide frequency, and sliding window analysis.
* **Full-Stack Application:** Implimented with a complete frontend, backend server for a seamless user experience.
* **Lightweight and scalable model:** This model has only 2.9 million parameters.

---

## Tech Stack

This full-stack application is built using:

| Category      | Technology                                                                                                                                                                                                                                             |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Frontend** | ![HTML](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white) ![SASS](https://img.shields.io/badge/SASS-hotpink.svg?style=for-the-badge&logo=SASS&logoColor=white) ![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E) |
| **Backend** | ![NodeJS](https://img.shields.io/badge/node.js-6DA55F?style=for-the-badge&logo=node.js&logoColor=white) ![Express.js](https://img.shields.io/badge/express.js-%23404d59.svg?style=for-the-badge&logo=express&logoColor=%2361DAFB) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
| **ML/DS** | ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) |
| **Database** | ![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white)                                                                                                                                         |
| **Deployment**| ![Render](https://img.shields.io/badge/Render-%46E3B7.svg?style=for-the-badge&logo=render&logoColor=white) ![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=white)           |

---

## Model Architecture

Here is the lightweight architecture of the model.

![Architecture](<./Node Server/public/images/Slide 16_9 - 1.jpg>)

## Model Performance

The model's performance is summarized in the classification report below:

|               | Precision | Recall | F1-Score |
| :------------ | :-------: | :----: | :------: |
| 0             |   0.87    |  0.87  |   0.87   |
| 1             |   0.85    |  0.86  |   0.85   |
|               |           |        |          |
| **Accuracy**  |     -     |   -    | **0.87** |
| **Macro Avg** |   0.86    |  0.86  |   0.86   |
| **Weighted Avg**|   0.87    |  0.87  |   0.87   |

## Getting Started: Local Installation

To get a local copy up and running, follow these simple steps.

### Prerequisites

You need to have Node.js and Python installed on your machine.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/Ankit6174/Domain-Specific-Mini-LLM-for-Genomic-Mutation-Prediction.git](https://github.com/Ankit6174/Domain-Specific-Mini-LLM-for-Genomic-Mutation-Prediction.git)
    cd Domain-Specific-Mini-LLM-for-Genomic-Mutation-Prediction
    ```

2.  **Set up the Node Server (Frontend & API):**
    ```sh
    cd "Node Server"
    npm install
    ```
    This will start the frontend server, typically on `http://localhost:8000`.

3.  **Set up the Python Server ML Model (Optional):**
    ```sh
    git lfs install
    git clone https://huggingface.co/spaces/ankitt6174/dna-mutation-prediction 
    
    # If you want to clone without large files - just their pointers
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/ankitt6174/dna-mutation-prediction
    ```
    Or checkout my huggingface account - [spaces/ankitt6174/dna-mutation-prediction](https://huggingface.co/spaces/ankitt6174/dna-mutation-prediction)

4. **Documentation**: To get the prediction you have to keep some constraints in your mind:-

    - **DNA Sequence**: The sequence must be exactly **199 base pairs** long. For example, if you want to analyze the significance of a mutation, you should extract a segment consisting of **100 nucleotides left** of the mutation site + the **mutated nucleotide itself** + **100 nucleotides right** of it.

    - **Reference and Alternate Alleles**: Both the reference and alternate alleles must be single nucleotide bases, represented by one of the letters **'A'**, **'T'**, **'G'**, or **'C'**.

    - **Mutation Type**: You can choose it my simple dropdown input.

    - **Chromosome**: It must the in the range of `chr1` - `chr12`. Because our model in trained on only these chromosome.

    - **Genomic Position**: Must be a number.

    Here are some sample data so you can evaluate:
    
    | Sequence | Label | Ref | Alt | Mutation | Chromosome | Position |
    |-----------|--------|-----|-----|-----------|-------------|-----------|
    | CCGGGCCAGCTGGTGGGCCGCATCATCCAGAAAGACCCCCTGCGCCGCTTCGATGGCTACCTCAACCAGGGCGCCAACAACAAGAAGATTGCCAAGGATGTCTTCAAGAAGGGGGACCAGGCCTACCTTACTGGTGGGTCCCCAGCCCTTCACAGCCCCTTCCTGAGGGTTGGGGGAGGAGGGGACCTTCTCCCACCTCCA | Pathogenic | T | A | T→A | chr9 | 128355158 |
    | TCTTCCAGAAGAGGCAGAAGCTTATCTACTTCAAATTCTGCCTCATACAATACGAGGAGGTGCAGAAGTGAGCGAGCCAGCGGAGGTATAACCCTTGTTATGCTTTATGCTTGTTAATATTTCTGTGCTGTACAAGAACCAAAGAATTTGGTCATACACATCAACTTTATGTTTAGTTTGAAGGGACAGGAAAAACACTGG | Benign | T | C | T→C | chr5 | 90628848 |
    | TGGGTAAAGAGAGAGCTCGTAGCTCCAGGGATCCCCCAGGCAGAGGCTGTGTGCACCTGTCCCTCTTCTTGCAGGGATCCCGCCGGCCTGACAGGCCCCCACCTCACCTTGGAGGCGGCCCAGTCCATCCAGCCTTCCGCACAAGGGTTCACGTTGATAAGGACAAGGCCCTCCACCATCTCAGGGTTGTTTAGCTGCAAT | Benign | A | C | A→C | chr8 | 133256770 |
    | TCATCGGTTATATGATGAAGATATTGTACGAGACAGTGGACATATTGTTAAATGTTTAGATTCTTTTTGTGATCCATTTCTCATTTCTGATGAGTTACGAAGAGTAAGTACAGAATTTTAAAATTTCGCAGTGGAATTTTATTTCAGGAAATGCAAAAATGGCAAAAACTTTGCCTGCTTAAAGCATAAAATACCTTGTCA | Pathogenic | A | T | A→T | chr11 | 102066649 |
    | CATGGGAGAGGGGGTGCCCACAAATGCACCCGTCCCAGGGGACCGGACAAAATTGGGGGGCTGCCCACTTGGGACCTTGGCATGGAGCTCACCTGCTGGCCCCGCGGGCAGGGCTGCTGGGAACCCCCCAGCCCCCAGCGAAGTGTGGGCTAGAGACCCAGCCTTAAAGGCAACTTCAGGGGGCTGGGGTCGGGGTGGCTT | Benign | C | T | C→T | chr12 | 49040268 |
    | CCCAGTGCCCTAATATCTGACCCCAGGTCCCTCGCCCTTCAACATTAGGCCTTCCTGACCAGAAAAAAACCAATCTTGTTTCTTTCCTACCTTGAGGCCCCGGGGACCCATGAAGCCAACATCTCCTTTTTCTCCTCGGATACCAGGCACTCCATCCTTTCCTGGGGATCCCTAGCAGGGAGAGGGTCCATGTGAGGTCAG | Pathogenic | C | G | C→G | chr3 | 48568100 |
    | TATTTGTAAGCTGAACTATCATTTTAACACAAAAGCTATCATTCTTGCTCAATGGAGTCAGGCTGCTCTTGGAGTTTCTGTCCTGGGAGGAAAAAGGGCAGGGTGTAGGTACCTGATGGTTTTCCACACGTCGAAGCCATCCAGAGGCTTTGTGCCATTGGTGTGTCCCCTGGCCAGCTTCACGAGTGTTGGCAGCCAGTC | Benign | G | A | G→A | chr5 | 78885572 |
    | AAGACATTAATGGATGCTCATACTGTCAATACAAAATCGAAAGTACCTGTTAGCATTCTAAGCTTCTATGTACTATACCTCTCATTTAAAATGTTACTTACAGATATTTTGCTACTTTCTGGTACTGCTTCATCACTGAAAGTGTCATTTGTTTCTATATCCATCCTTGGCCTTTTTCTAACATTGACATCTTCCTCCTGT | Pathogenic | C | T | C→T | chr8 | 89953243 |
    | CTGTGGTCGGCTACAACCCAGAAAAATACTCAGTGACACAGCTCGTTTACAGCGCCTCCATGGACCAGATAAGTGCCATCACTGACAGTGCCGAGTACTGCGAGCAGTATGTCTCCTATTTCTGCAAGATGTCAAGATTGTTGAACACCCCAGGTAGGCTGAGAATGGAATGTTACTTTTAATCACTATCTCAGCTGGTGC | Benign | C | T | C→T | chr7 | 147639254 |
    | TCCAATGGGATTCTGCCTATATTCTGTCCAGATCAAAGTTTTTACAAATTGAATTTTACAATGAAATGAAAAATAAAATCTCAAGTGTACTTACCAAACTGCATGCCCCAATCGCCAAGGTAATTTATTCTTATTACTTGATGTCCTAAAGCTTCTTTGAGATTTGCTATAAAATTTCCTAGTAATCAATAAAGTATATAG | Pathogenic | G | A | G→A | chr6 | 87545622 |

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## Author

MANASV JAIN

* [Twitter](https://x.com/Ankit6174)
* [Medium](https://medium.com/@ankit6174)
* [LinkedIn](https://www.linkedin.com/in/Ankit6174) 
