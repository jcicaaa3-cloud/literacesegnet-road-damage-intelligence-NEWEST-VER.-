# Asset and license policy

This repository is prepared for public portfolio use. Keep it clean before pushing to GitHub.

## Safe to commit

- source code written for this project
- configuration files
- small placeholder files such as `.gitkeep`
- documentation
- architecture diagrams authored for this project
- result templates without real private data

## Do not commit

- raw road images
- dataset masks or labels
- downloaded public datasets
- paid or restricted datasets
- private camera footage
- trained checkpoints
- pretrained model weights
- Hugging Face cache folders
- generated overlays if the input image license is unclear
- `.env`, API keys, cloud credentials
- thesis DOCX/PDF drafts containing personal information or school formatting

## MIT scope

`LICENSE` covers only this repository's own code and documentation. It does not cover third-party packages, model weights, datasets, or APIs.

When GitHub displays this project as MIT-licensed, read that as:

> The code written for this repository is MIT-licensed. External assets are not included and are not relicensed.

## Dataset wording for README or resume

Use this wording:

> The repository does not include datasets or weights. Experiments require the user to place permitted image-mask pairs under the documented dataset layout.

Avoid this wording:

> Dataset is included.
> We provide pretrained weights.
> Anyone can freely use all assets in this repository.

## Result image policy

Only publish overlay images when the source road image can be redistributed. If the license is unclear, keep overlays local and publish only aggregate metrics.