from dependency_injector import containers, providers
from chemnlp.data_processing.template_sampler import TemplateSampler
from chemnlp.data.utils import load_yaml
import pandas as pd
import logging

class Container(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["config/default_config.yaml"])

    # Configure logging
    logging = providers.Resource(
        logging.basicConfig,
        level=config.logging.level,
        format=config.logging.format
    )

    # Provide the logger
    logger = providers.Factory(logging.getLogger, name="chemnlp")

    # Provide the YAML loader
    yaml_loader = providers.Factory(load_yaml)

    # Provide the DataFrame loader
    df_loader = providers.Factory(
        pd.read_csv,
        low_memory=False
    )

    # Provide the TemplateSampler
    template_sampler = providers.Factory(
        TemplateSampler,
        logger=logger,
        yaml_loader=yaml_loader,
        df_loader=df_loader,
        config=config.template_sampler
    )
