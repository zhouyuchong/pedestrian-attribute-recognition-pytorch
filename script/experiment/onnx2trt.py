import tensorrt as trt
import os
TRT_LOGGER = trt.Logger()
def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network() as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30
            # builder.max_workspace_size = 1 << 30 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            print('network layers ',network.num_layers)

            # engine = builder.build_cuda_engine(network)
            plan = builder.build_serialized_network(network, config)
            with trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

if __name__ == '__main__':
    model_path = 'peta.onnx'
    trt_engine = 'deepMAR.trt'
    get_engine(model_path,trt_engine)