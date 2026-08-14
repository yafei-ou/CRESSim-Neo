# Contributing to CRESSim-Neo

Thanks for your interest in contributing to CRESSim-Neo.

Before a contribution can be accepted, CRESSim-Neo must have a signed
Contributor License Agreement (CLA) on file for every contributor. Choose and
download the agreement that applies to your contribution:

- [Individual CLA](CLA/CRESSim-Neo-Individual-CLA.pdf) — for contributions
  made in your individual capacity.
- [Entity CLA](CLA/CRESSim-Neo-Entity-CLA.pdf) — for contributions made on
  behalf of a company or other legal entity.

Sign the applicable agreement and email the completed PDF to the project
maintainer [yafei+cressim-neo@yafei.dev](mailto:yafei+cressim-neo@yafei.dev).
Use the subject `CRESSim-Neo CLA: <GitHub username> <legal name>`. The GitHub
username must be the account you will use to open pull requests.

You only need to submit a CLA once. A pull request cannot be merged until the
maintainer has received the signed agreement.

## Pull requests

Submit code and documentation changes as GitHub pull requests. Please make
sure that your pull request:

- merges cleanly with the current default branch;
- is rebased on the current default branch and does not introduce merge commits;
- includes focused tests when the change affects behavior;
- passes the relevant test suite; and
- follows the style already established in the files you change. Use the
  repository's `.clang-format` configuration for new or modified C++ code.

The main branch is kept linear. Rebase your branch before it is merged; do not
merge the default branch into your pull-request branch.

For a typical CMake build and test workflow, see the [development
instructions](README.md#native-development) in the README.
