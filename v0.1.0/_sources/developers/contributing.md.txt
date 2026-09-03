# Contributing

All contributions require a Contributor License Agreement (CLA) before a pull
request can be merged. Submit one agreement once per contributor:

- [Individual CLA](../../../CLA/CRESSim-Neo-Individual-CLA.pdf) for work
  contributed in an individual capacity.
- [Entity CLA](../../../CLA/CRESSim-Neo-Entity-CLA.pdf) for work contributed
  on behalf of a company or other legal entity.

Email the completed PDF to
[yafei+cressim-neo@yafei.dev](mailto:yafei+cressim-neo@yafei.dev) with the
subject `CRESSim-Neo CLA: <GitHub username> <legal name>`. Use the GitHub
account that will open the pull request.

## Pull requests

Submit code and documentation changes as GitHub pull requests. Before opening
or updating one:

- Rebase it on the current default branch; do not merge the default branch into
  the pull-request branch.
- Keep the change focused and make sure it merges cleanly.
- Add focused tests when behavior changes and run the relevant test profile.
- Follow the style of the files you modify. Format modified C++ with the
  repository's `.clang-format` configuration.

Use {doc}`testing` for verification commands and {doc}`../getting-started/build`
for local build setup. The repository-facing policy is maintained in the root
`CONTRIBUTING.md` file.
