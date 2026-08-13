# Contributing to CRESSim-Neo

Thanks for your interest in contributing to CRESSim-Neo.

Before a contribution can be accepted, CRESSim-Neo must have a signed
Contributor License Agreement (CLA) on file for every contributor. Download the
project CLA [`CLA.pdf`](CLA.pdf), sign it, and email the completed agreement
to the project maintainer [yafei+cressim-neo@yafei.dev](mailto:yafei+cressim-neo@yafei.dev).
Attach the signed PDF and use the subject `CRESSim-Neo CLA: <GitHub username> <legal name>`.
The GitHub username must be the account you will use to open pull requests.

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
