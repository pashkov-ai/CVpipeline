# Textile Fabric Database

## Dataset Source

**Kaggle**: [AITEX Fabric Image Database](https://www.kaggle.com/datasets/nexuswho/aitex-fabric-image-database)

## Dataset Overview

The textile fabric database consists of **245 images** of **7 different fabric types**:
- **140 defect-free images** (20 per fabric type)
- **105 defective images** (with various defect types)

## Image Specifications

- **Resolution**: 4096 × 256 pixels

## File Naming Convention

### Defective Images
Format: `nnnn_ddd_ff.png`
- `nnnn` = Image number
- `ddd` = Defect code
- `ff` = Fabric code

### Defect Masks
Format: `nnnn_ddd_ff_mask.png`
- White pixels represent the defect area

### Defect-Free Images
Format: `nnnn_000_ff.png`
- Defect code is replaced with `000`

## Defect Types

| Defect Code | Defect Description |
|-------------|-------------------|
| 2           | Broken end        |
| 6           | Broken yarn       |
| 10          | Broken pick       |
| 16          | Weft curling      |
| 19          | Fuzzyball         |
| 22          | Cut selvage       |
| 23          | Crease            |
| 25          | Warp ball         |
| 27          | Knots             |
| 29          | Contamination     |
| 30          | Nep               |
| 36          | Weft crack        |

## Citation

**Title**: AFID: A Public Fabric Image Database for Defect Detection

**Authors**: Javier Silvestre-Blanes, Teresa Albero-Albero, Ignacio Miralles, Rubén Pérez-Llorens, Jorge Moreno

**Publication**: AUTEX Research Journal, No. 4, 2019

**Paper**: [https://content.sciendo.com/view/journals/aut/ahead-of-print/article-10.2478-aut-2019-0035.xml](https://content.sciendo.com/view/journals/aut/ahead-of-print/article-10.2478-aut-2019-0035.xml)