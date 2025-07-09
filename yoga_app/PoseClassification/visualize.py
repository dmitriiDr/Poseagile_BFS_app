import io
from PIL import Image, ImageFont, ImageDraw
import requests
import matplotlib.pyplot as plt


class PoseClassificationVisualizer(object):
    def __init__(
        self,
        plot_location_x=0.05,
        plot_location_y=0.05,
        plot_max_width=0.4,
        plot_max_height=0.4,
        plot_figsize=(9, 4),
        plot_x_max=None,
        plot_y_max=None,
        counter_location_x=0.95,
        counter_location_y=0.05,
        counter_font_path="https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Regular.ttf?raw=true",
        counter_font_color="red",
        counter_font_size=0.05,
    ):
        self._plot_location_x = plot_location_x
        self._plot_location_y = plot_location_y
        self._plot_max_width = plot_max_width
        self._plot_max_height = plot_max_height
        self._plot_figsize = plot_figsize
        self._plot_x_max = plot_x_max
        self._plot_y_max = plot_y_max
        self._counter_location_x = counter_location_x
        self._counter_location_y = counter_location_y
        self._counter_font_path = counter_font_path
        self._counter_font_color = counter_font_color
        self._counter_font_size = counter_font_size

        self._counter_font = None
        self._pose_classification_history = []
        self._pose_classification_filtered_history = []
        self._pose_hold_frame_count = 0
        self._pose_hold_duration_sec = 0
        self._fps = None
        self._last_output_height = None

    def __call__(
        self,
        frame,
        pose_classification,
        pose_classification_filtered,
        repetitions_count,
        top_class_name=None
    ):
        self._pose_classification_history.append(pose_classification)
        self._pose_classification_filtered_history.append(pose_classification_filtered)

        # Duration tracking
        confidence_threshold = 6
        current_confidence = 0
        if pose_classification_filtered is not None and top_class_name:
            current_confidence = pose_classification_filtered.get(top_class_name, 0)

        if current_confidence > confidence_threshold:
            self._pose_hold_frame_count += 1
        else:
            self._pose_hold_frame_count = 0

        if self._fps:
            self._pose_hold_duration_sec = self._pose_hold_frame_count / self._fps

        output_img = Image.fromarray(frame)
        output_width, output_height = output_img.size

        # Draw classification plot
        # if top_class_name:
        #     img = self._plot_classification_history(output_width, output_height, top_class_name)
        #     img.thumbnail(
        #         (
        #             int(output_width * self._plot_max_width),
        #             int(output_height * self._plot_max_height),
        #         ),
        #         Image.LANCZOS,
        #     )
        #     output_img.paste(
        #         img,
        #         (
        #             int(output_width * self._plot_location_x),
        #             int(output_height * self._plot_location_y),
        #         ),
        #     )

        # Draw text
        output_img_draw = ImageDraw.Draw(output_img)
        font_size = int(output_height * self._counter_font_size)
        if self._counter_font is None:
            # font_size = int(output_height * self._counter_font_size)
            font_request = requests.get(self._counter_font_path, allow_redirects=True)
            self._counter_font = ImageFont.truetype(io.BytesIO(font_request.content), size=font_size)
            self._last_output_height = output_height

        if self._fps:
            duration_text = f"{self._pose_hold_duration_sec:.1f}s"
            output_img_draw.text(
                (
                    output_width * self._counter_location_x * 0.85,
                    output_height * (self._counter_location_y),
                ),
                duration_text,
                font=self._counter_font,
                fill="blue",
            )

        if top_class_name and current_confidence > confidence_threshold:
            output_img_draw.text(
                (
                    output_width * self._counter_location_x * 0.05,
                    output_height * self._counter_location_y,
                ),
                f'Pose: {top_class_name}',
                font=self._counter_font,
                fill=self._counter_font_color,
            )

        return output_img

    def _plot_classification_history(self, output_width, output_height, class_name):
        fig = plt.figure(figsize=self._plot_figsize)

        for classification_history in [
            self._pose_classification_history,
            self._pose_classification_filtered_history,
        ]:
            y = []
            for classification in classification_history:
                if classification is None:
                    y.append(None)
                elif class_name in classification:
                    y.append(classification[class_name])
                else:
                    y.append(0)
            plt.plot(y, linewidth=7)

        plt.grid(axis="y", alpha=0.75)
        plt.xlabel("Frame")
        plt.ylabel("Confidence")
        plt.title(f"Classification history for `{class_name}`")
        plt.legend(loc="upper right")

        if self._plot_y_max is not None:
            plt.ylim(top=self._plot_y_max)
        if self._plot_x_max is not None:
            plt.xlim(right=self._plot_x_max)

        buf = io.BytesIO()
        dpi = min(
            output_width * self._plot_max_width / float(self._plot_figsize[0]),
            output_height * self._plot_max_height / float(self._plot_figsize[1]),
        )
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        return img

