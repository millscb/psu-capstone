from datetime import datetime

from htmrl.encoder_layer.base_encoder import BaseEncoder
from htmrl.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from htmrl.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from htmrl.sdr_layer.sdr import SDR

if __name__ == "__main__":

    test_date = datetime(2023, 3, 15, 14, 30)  # March 15, 2023, 14:30

    params = DateEncoderParameters()

    encoder = params.encoder_class(params)

    sdr = encoder.encode(test_date)

    print(f"Date Encoder Size: {encoder.size}")
    print(f"SDR: {sdr}")
