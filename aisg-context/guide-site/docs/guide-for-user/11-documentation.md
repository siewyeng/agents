# Documentation

The boilerplate packages generated by the template are populated with
some
[NumPy formatted docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).
What we can do with this is to observe how documentation can be
automatically generated using
[Sphinx](https://www.sphinx-doc.org/en/master/),
with the aid of the
[Napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
extension. Let's build the HTML asset for the documentation:

```bash
# From the root folder
$ conda activate stems-mini-project
$ sphinx-apidoc -f -o docs src
$ sphinx-build -b html docs public
```

Open the file `public/index.html` with your browser and you will be
presented with a static site similar to the one shown below:

![Sphinx - Generated Landing Page for Documentation Site](assets/screenshots/sphinx-generated-doc-landing-page.png)

Browse through the site and inspect the documentation that was
automatically generated through Sphinx.

## GitLab Pages

Documentation generated through Sphinx can be served on
[GitLab Pages](https://docs.gitlab.com/ee/user/project/pages/), through
GitLab CI/CD. With this template, a default CI job has been defined
in `.gitlab-ci.yml` to serve the Sphinx documentation when pushes are
done to the `main` branch:

```yaml
...
pages:
  stage: deploy-docs
  image:
    name: continuumio/miniconda:4.7.12
  script:
  - conda env update -f stems-mini-project-conda-env.yaml
  - conda init bash
  - source ~/.bashrc
  - conda activate stems-mini-project
  - sphinx-apidoc -f -o docs src
  - sphinx-build -b html docs public
  artifacts:
    paths:
    - public
  only:
  - main
...
```

The documentation page is viewable through the following convention:
`<NAMESPACE>.gitlab.aisingapore.net/<PROJECT_NAME>` or
`<NAMESPACE>.gitlab.aisingapore.net/<GROUP>/<PROJECT_NAME>`.

__Reference(s):__

- [GitLab Docs - Pages domain names, URLs, and base URLs](https://docs.gitlab.com/ee/user/project/pages/getting_started_part_one.html)
- [GitLab Docs - Namespaces](https://docs.gitlab.com/ee/user/group/#namespaces)
