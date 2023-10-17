# STEMSS Mini Project

![AI Singapore's Kapitan Hull EPTG Onprem Run:ai Banner](./aisg-context/guide-site/docs/assets/images/kapitan-hull-eptg-onprem-runai-banner.png)

## Description
In this repository, we created 2 AI agents with the Vertex AI model. It is built upon the `langchain_experimental` library and deployment can be done on `streamlit`.

## Deployment

The code can easily deployed through running in the root directory of this repository:

```streamlit run streamlit_app.py```

We recommend creating a new environment for this project by executing the following command in the root directory

```conda env create --name <ENVIRONMENTNAME> --file=conda.yml```

Alternatively, you can build a docker image by running the command:

=== "Windows PowerShell"

    ```
    $ docker build `
        -t registry.aisingapore.net/aiap-14-dsp/stems-mini-project/streamlit-app:0.1.0 `
        -f docker/streamlit-app.Dockerfile .
    ```

### Navigating Streamlit
Using the navigation sidebar in the image below, follow the order of the menu and configure the different hyperparameters for the model.

![alt text](./img/streamlit_menu.png)

After going through the first 3 buttons on the menu, the agents can now be initialised by clicking the "Generate Agents" button

![alt text](./img/agent_initilisation.png)

Finally, we can interact with any agent and even choose to inject memories to contextualise the interactions.

![alt text](./img/agent_interaction.png)



## Other information
__Customised for `STEMSS Mini Project`__.

__Project Description:__ Creation of 2 interactive AI agents with Vertex AI LLM 

This template that is also accompanied with an end-to-end guide was
generated and customised using the
following
[`cookiecutter`](https://cookiecutter.readthedocs.io/en/stable/)
template:
https://github.com/aisingapore/ml-project-cookiecutter-onprem-runai

The contents of the guide have been customised
according to the inputs provided upon generation of this repository
through the usage of `cookiecutter` CLI,
following instructions detailed
[here](https://github.com/aisingapore/ml-project-cookiecutter-onprem-runai/blob/main/README.md)
.

Inputs provided to `cookiecutter` for the generation of this
template:

- __`project_name`:__ STEMSS Mini Project
- __`description`:__ Creation of 2 interactive AI agents with Vertex AI LLM 
- __`repo_name`:__ stems-mini-project
- __`src_package_name`:__ stems_mini_project
- __`src_package_name_short`:__ smp
- __`author_name`:__ STEMSS
- __`harbor_registry_project_path`:__ registry.aisingapore.net/aiap-14-dsp/stems-mini-project


