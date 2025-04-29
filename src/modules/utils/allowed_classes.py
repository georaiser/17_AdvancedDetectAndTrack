# -*- coding: utf-8 -*-
class AllowedClasses:
    def __init__(self, config):
               
        classes = config.sub_configs.get("classes")["classes"]
        model_name = config.get("processor", "model")

        if "yolo" in model_name:
            self.class_names = classes["coco_80"]
        elif "faster" or "efficientdet" in model_name:
            self.class_names = classes["coco_91"]
        else:
            print("Model not supported")
            exit(0)

    def get_allowed_classes(self):
        self.allowed_classes = [
            self.class_names.index("person"),
            self.class_names.index("car"),
            self.class_names.index("truck"),
            self.class_names.index("bus"),
        ]

        # self.allowed_classes = [
        #     self.class_names.index("suitcase"),
        #     self.class_names.index("backpack"),
        #     self.class_names.index("handbag"),
        # ]

        # self.allowed_classes = list(range(0, len(self.class_names)))

        return self.class_names, self.allowed_classes
