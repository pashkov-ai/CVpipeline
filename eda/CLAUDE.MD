## Folder Organization
- Use notebooks (`.ipynb`) for exploratory analysis only
- Keep helper functions in dedicated `helper_*.py` files
- Move validated code to `.py` modules

## Conda Environment
- Always use `CVpipeline-eda` **conda** for package management (configured for this project) in this directory
- Check `environment.nobuilds.yml` before adding dependencies
- Document why new packages are needed