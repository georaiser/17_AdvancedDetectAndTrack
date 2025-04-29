from PIL import Image
import torch
import numpy as np
import cv2


class ImageSquare:
    @staticmethod
    def calculate_dimensions(original_width, original_height, target_size):
        """Calculate new dimensions maintaining aspect ratio"""
        if original_width > original_height:
            new_height = int(target_size * original_height / original_width)
            new_width = target_size
            top_pad = (target_size - new_height) // 2
            bottom_pad = target_size - new_height - top_pad
            left_pad = right_pad = 0

        else:
            new_width = int(target_size * original_width / original_height)
            new_height = target_size
            left_pad = (target_size - new_width) // 2
            right_pad = target_size - new_width - left_pad
            top_pad = bottom_pad = 0

        padding_info = {
            "top_pad": top_pad,
            "bottom_pad": bottom_pad,
            "left_pad": left_pad,
            "right_pad": right_pad,
            "original_size": (original_width, original_height),
            "resized_size": (new_width, new_height),
            "padded_size": (target_size, target_size),
        }

        return padding_info

    @staticmethod
    def pad_to_square(image, padding_info):
        if isinstance(image, Image.Image):
            # Resize PIL image to maintain aspect ratio
            resized_image = image.resize(padding_info["resized_size"])
            # Create new image with padding
            padded_image = Image.new("RGB", (padding_info["padded_size"]), (0, 0, 0))
            padded_image.paste(
                resized_image, (padding_info["left_pad"], padding_info["top_pad"])
            )
        elif isinstance(image, np.ndarray):
            # Resize numpy array image maintaining aspect ratio
            resized_image = cv2.resize(image, padding_info["resized_size"])
            # Create padded image
            padded_image = np.zeros(
                (padding_info["padded_size"][1], padding_info["padded_size"][0], 3),
                dtype=np.uint8,
            )
            # Place resized image in padded image
            padded_image[
                padding_info["top_pad"] : padding_info["top_pad"]
                + resized_image.shape[0],
                padding_info["left_pad"] : padding_info["left_pad"]
                + resized_image.shape[1],
            ] = resized_image

            # Convert numpy array to PIL image
            padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

        else:
            raise TypeError("Image must be either PIL Image or NumPy array")

        return resized_image, padded_image

    @staticmethod
    def unpad_coordinates(coords, padding_info):
        """Adjusts boxes coordinates from padded space to original/resized space."""
        if len(coords) == 0:
            return coords

        # Convert numpy array to torch tensor if needed
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)

        # Extract padding information
        top_pad = padding_info["top_pad"]
        left_pad = padding_info["left_pad"]
        padded_size = padding_info["padded_size"]
        # output_size = padding_info["resized_size"]
        output_size = padding_info["original_size"]

        # Remove padding
        adjusted_coords = coords.clone()
        adjusted_coords[:, [0, 2]] -= left_pad  # x coordinates
        adjusted_coords[:, [1, 3]] -= top_pad  # y coordinates

        # Scale to resized dimensions
        scale_x = output_size[0] / (padded_size[0] - 2 * left_pad)
        scale_y = output_size[1] / (padded_size[1] - 2 * top_pad)

        adjusted_coords[:, [0, 2]] *= scale_x
        adjusted_coords[:, [1, 3]] *= scale_y

        return adjusted_coords
