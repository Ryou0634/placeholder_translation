import logging
import click

import commands


@click.group()
@click.option("--verbose", is_flag=True)
def cli(verbose: bool):
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt)
        logging.getLogger("transformers").setLevel(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt)
        logging.getLogger("transformers").setLevel(level=logging.WARNING)


cli.add_command(commands.train_spm_from_config)
cli.add_command(commands.translate)

if __name__ == "__main__":
    cli()
