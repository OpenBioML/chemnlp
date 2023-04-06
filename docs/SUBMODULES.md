# Introduction
This page outlines the workflow for contributing to the ChemNLP project where changes to the Git submodules are required. The project currently has two submodules:

1. [gpt-neox](https://github.com/OpenBioML/gpt-neox)
2. [lm-eval2](https://github.com/OpenBioML/lm-eval2)

where both of these are forks from [EleutherAI](https://github.com/EleutherAI).

# What are git submodules?

Submodules allow us to keep seperate Git repositories as subdirectories inside ChemNLP. As these submodules are forks we can both make any changes we require to them (and pin a specific commit) as well as periodically integrate changes from the original upstream (EleutherAI) repository.

You can think of both the `gpt-neox` and `lm-eval2` submodules as separate Git repositories with their own remotes, commit history and branches etc...

In essence, all the ChemNLP project does is to track which commit we are using for each submodule (to see this run `git submodule status` from `chemnlp`).

There are many excellent introductions to submodules online and we won't repeat them here. Instead we'll outline the process for working with them on the ChemNLP project and we encourage you to read more about them if of interest. Here are some links you might find useful:

1. [7.11 Git Tools - Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) - section from Pro Git.
2. [Git submodule docs](https://git-scm.com/docs/git-submodule) - the documentation.


# Getting help
The instructions below attempt to guide you through the process of working with submodules. However, if you are still confused please reach out on GitHub or Discord to a project maintainer.

# Workflow 1: making changes to a submodule only

Example of making a change to the `gpt-neox` submodule for a feature called `add-peft-method`.

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the [ChemNLP repository](https://github.com/OpenBioML/chemnlp) from your personal GitHub account.
2. [Clone your fork](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) and the submodules, see: [Cloning submodules](../README.md#cloning-submodules).
3. [Optional, if required for the issue] Install `chemnlp` in your virtual env using `pip install -e` (see installation instructions [here](../README.md#installation-and-set-up)).
4. [Make a new branch](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) e.g. `feat(sub):add-peft-method` in the `gpt-neox` submodule, **not** in `chemnlp`.
6. Make changes to the `gpt-neox` submodule per the issue you are working on.
7. Commit changes in the `gpt-neox` submodule.
8. Push the submodule changes to remote and open a PR in [gpt-neox](https://github.com/OpenBioML/gpt-neox).
10. Once the changes to the submodule are approved, merge them (or a reviewer will).

The above **only** updates the `gpt-neox` submodule on remote - it **does not** change which commit `chemnlp` is tracking. To do this:

1. On your fork of `chemnlp`, update to get the latest changes for the `gpt-neox` submodule only: `git submodule update --remote gpt-neox`
2. This will checkout the latest commit on the `main` branch of `gpt-neox`.
   -  Note: if you want to track a different commit of `gpt-neox` other than the latest then navigate to the `gpt-neox` directory and checkout a specific commit (e.g. your recent merge commit from the `gpt-neox` pull request above): `git checkout <commit-hash>`
3. In `chemnlp` [make a new branch](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) e.g. `feat:update-gpt-neox-submodule`
4. Commit this change, push to your fork's remote and open a PR from your fork to the [ChemNLP repository](https://github.com/OpenBioML/chemnlp) which will update the commit the `chemnlp` project tracks.


Things to note:

* The remote of `chemnlp` should be your fork.
* The remote of `gpt-neox` should be the [OpenBioML fork](https://github.com/OpenBioML/gpt-neox).

To see the remotes for a Git repository run: `git remote -v`

# Workflow 2: making changes to both ChemNLP and a submodule

If you need to make changes to the main `chemnlp` project at the same time as a submodule the above workflow can be modified to accomodate this. It's advisable to make changes to the submodule first then once these are merged, submit a PR to the [ChemNLP repository](https://github.com/OpenBioML/chemnlp) which (i) adds changes to `chemnlp` and (ii) updates the `gpt-neox` commit which `chemnlp` tracks.

# Appendix

## Detached HEADs & submodules

Usually, when working with Git, you have a certain *branch* checked out. However, Git also allows you to check out any arbitrary commit. Working in such a non-branch scenario is called having a "detached HEAD".

With submodules: using the `update` command (e.g. `git submodule update`) on a submodule *checks out a specific commit - not a branch*. This means that the submodule repository will be in a "detached HEAD" state.

🚨 **Don't commit on a detached HEAD** 🚨

When you work in the submodule directly you should create or checkout a branch before committing your work.

See also: [why did Git detach my HEAD?](https://stackoverflow.com/questions/3965676/why-did-my-git-repo-enter-a-detached-head-state/3965714#3965714)

> Any checkout of a commit that is not the name of one of *your* branches will get you a detached HEAD. A SHA1 which represents the tip of a branch still gives a detached HEAD. Only a checkout of a local branch *name* avoids that mode.
>
