def _require_cycnn_extension(model_name: str):
    raise RuntimeError(
        f"{model_name} requires the CUDA-only CyConv2d extension. "
        "Use the GPU Docker image for CyCNN models, or choose a classic CPU model "
        "such as vgg19 or resnet20."
    )


def get_model(model, dataset, classify=True):
    """Return a model by name.

    Classic CNN models can run in the CPU image.
    CyCNN models require the GPU image because they depend on CyConv2d_cuda.
    """

    if model in {"vgg11", "vgg13", "vgg16", "vgg19"}:
        import models.vgg as vgg

        constructors = {
            "vgg11": vgg.vgg11_bn,
            "vgg13": vgg.vgg13_bn,
            "vgg16": vgg.vgg16_bn,
            "vgg19": vgg.vgg19_bn,
        }
        return constructors[model](dataset=dataset, classify=classify)

    if model in {"resnet20", "resnet32", "resnet44", "resnet56"}:
        import models.resnet as resnet

        constructors = {
            "resnet20": resnet.resnet20,
            "resnet32": resnet.resnet32,
            "resnet44": resnet.resnet44,
            "resnet56": resnet.resnet56,
        }
        return constructors[model](dataset=dataset)

    if model in {"cyvgg11", "cyvgg13", "cyvgg16", "cyvgg19"}:
        try:
            import models.cyvgg as cyvgg
        except ModuleNotFoundError as exc:
            if exc.name == "CyConv2d_cuda":
                _require_cycnn_extension(model)
            raise

        constructors = {
            "cyvgg11": cyvgg.cyvgg11_bn,
            "cyvgg13": cyvgg.cyvgg13_bn,
            "cyvgg16": cyvgg.cyvgg16_bn,
            "cyvgg19": cyvgg.cyvgg19_bn,
        }
        return constructors[model](dataset=dataset, classify=classify)

    if model in {
        "cyresnet20",
        "cyresnet32",
        "cyresnet44",
        "cyresnet56",
        "cyresnet110",
        "cyresnet1202",
    }:
        try:
            import models.cyresnet as cyresnet
        except ModuleNotFoundError as exc:
            if exc.name == "CyConv2d_cuda":
                _require_cycnn_extension(model)
            raise

        constructors = {
            "cyresnet20": cyresnet.cyresnet20,
            "cyresnet32": cyresnet.cyresnet32,
            "cyresnet44": cyresnet.cyresnet44,
            "cyresnet56": cyresnet.cyresnet56,
            "cyresnet110": cyresnet.cyresnet110,
            "cyresnet1202": cyresnet.cyresnet1202,
        }
        return constructors[model](dataset=dataset)

    raise ValueError(f"Unknown model: {model}")
