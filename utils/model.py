import os
import shutil
import torch
import wandb


def save_checkpoint(
    state,
    save_path,
    filename="checkpoint.pth.tar",
    is_initial=False,
    is_final=False,
    is_best=False,
    save_all=False,
    dummy=False,
):
    if dummy:
        return

    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_initial:
        file_path = os.path.join(save_path, "checkpoint_initial.pth.tar")
        shutil.copyfile(filename, file_path)

        try:
            wandb.save(file_path)
        except:
            pass

    if is_final:
        file_path = os.path.join(save_path, "checkpoint_final.pth.tar")
        shutil.copyfile(filename, file_path)

        try:
            wandb.save(file_path)
        except:
            pass

    if is_best:
        shutil.copyfile(filename, os.path.join(save_path, "checkpoint_best.pth.tar"))

    if save_all:
        shutil.copyfile(
            filename,
            os.path.join(
                save_path,
                f"checkpoint_task_{state['task']}_epoch_{state['epoch']}.pth.tar",
            ),
        )
