# Monocular Spacecraft Pose Estimation with Vision Transformers

**Instructions:**
- Create a folder named `datasets` into the folder `code`
- Download the speed dataset from https://zenodo.org/records/6327547 and put it in the folder `datasets`. You can also run inference on speed+ lab data by downloading the speed+ dataset from https://purl.stanford.edu/wv398fc4383 and recreating the same folder structure of speed
- Modify `code/src/config.py` with the desired configurations, including the desired model and either training or evaluation mode. The pre-trained models are available in the `code/models` folder
- from the folder `code/src` run `python main.py`

**Blog Post and Storyline:**
- You can find a blog post with a detailed description of the project's aim, experiments, and results at https://oasis-soul-ee5.notion.site/Monocular-Spacecraft-Pose-Estimation-with-Vision-Transformers-60d53328311a4b7482b2d4903b322126
- If you prefer a quicker overview of the work, you can find the project storyline in this repo as a pdf file

This work is an extension of the following paper:

Posso, J., Bois, G., & Savaria, Y. (2022). Mobile-URSONet: an Embeddable Neural Network for Onboard Spacecraft Pose Estimation. In *2022 IEEE International Symposium on Circuits and Systems (ISCAS)* (pp. 794-798). https://doi.org/10.1109/ISCAS48785.2022.9937721
