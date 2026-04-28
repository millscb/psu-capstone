import htmrl as htm


def main():

    logger = htm.log_tools.get_logger()
    logger.info("Starting HTMRL financial pipeline example.")
    print(f"HTMRL version: {htm.__version__}")

    # use the input tools to load the data
    ih = htm.input_tools.InputHandler()
    data = ih.input_data(htm.DATA_PATH)
    logger.info(f"Data loaded with shape: {len(data)}")

    # use the encoder tools to encode the data
    parameters = htm.encoder_tools.RDSEParameters(resolution=0.01, size=2048, sparsity=0.02)
    encoder = htm.encoder_tools.RandomDistributedScalarEncoder(parameters)

    # encoder2 = htm.encoder_tools.

    encoded_data_single = encoder.encode(data["Close"][0])  # encode the first value as an example
    logger.info(f"Data encoded into {len(encoded_data_single)} bits.")

    # show the graph of the encoded data
    htm.grapher_tools.plot_sdr(encoded_data_single, title="Encoded Financial Data")

    # use an Agent to process the data
    trainer = htm.agent_tools.Trainer(htm.agent_tools.Brain({}))
    fields = [("close_input", 2048, parameters)]
    brain = trainer.build_brain(fields=fields)  # build the brain with the specified fields

    trainer.train_column(
        {"close_input": data["Close"]}, 10, brain
    )  # train the column on the "Close" data for 100 steps
    trainer.show_active_columns(brain, "Financial Data")  # visualize active columns after training


if __name__ == "__main__":
    main()
