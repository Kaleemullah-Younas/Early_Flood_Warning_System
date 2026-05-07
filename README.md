**FloodSense — Neural Nova Data Drop**

Official dataset package for the *FloodSense* challenge, part of Neural Nova's 60-Hour Data Drop Sprint by BTech.

⚠️ Read This First

Open `data_dictionary.txt` before touching any data. It explains every column, known data issues, and how the datasets connect to each other. Skipping it will cost you time.

📁 Files

| File | Description |

| `floodsense_training_data.csv` | 1,434 daily flood sensor records across Pakistani districts (2022–2024). This is your primary training dataset. |

| `ndma_flood_impact_2022.csv` | NDMA regional impact data from the 2022 floods. Use for impact quantification in your pitch — not for ML training. |

| `district_elevation_reference.csv` | District-level average elevations from NASA SRTM. Merge on `district` column to use terrain as a feature. |

| `data_dictionary.txt` | Full column descriptions, known data issues, and usage guidance. Read before training. |

| `hackathon_notes.txt` | Challenge brief, deliverables, and judging context. |

🎯 The Challenge

- Build FloodSense — an AI-powered flood risk early warning system that predicts risk levels for Pakistani districts and presents them through a dashboard a non-technical district official can act on immediately.
- Model output: Low / Medium / High / Critical + confidence score  
- Target column: `flood_event` (0 = no flood, 1 = flood)


⚠️ Data Advisory

The dataset reflects real-world collection conditions. Teams are expected to inspect their data thoroughly before training.

- Some columns contain missing entries — your imputation choices should be documented and justifiable.
- At least one numeric column contains undefined or infinite values — handle these before passing data to your model.
- The dataset may contain redundant records — verify row uniqueness before splitting.
- Not all values in the dataset are trustworthy — anomalies exist and should be identified during exploration.
- Not all columns in the training CSV carry predictive signal — inspect feature variance before selecting inputs. A supplementary reference file is provided for certain geographic features.

📋 Judging Reminder

Judges are not looking for the highest accuracy score. They are looking for the team whose tool a district official in Nowshera could open on a slow laptop and actually use to make a decision that saves lives.


*Neural Nova — Build for a Better Tomorrow | BTech*
