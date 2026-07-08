// COCO class names, ported verbatim from the jax-js detr-resnet-50 demo.
// DETR/D-FINE use the original COCO category IDs, which have gaps ("N/A") for
// unused classes; index 0 is unused and the D-FINE obj2coco head emits labels
// in this same 0–90 space. See https://github.com/facebookresearch/detr/issues/108

export const COCO_CLASSES = [
  "N/A", // 0 - unused
  "person", // 1
  "bicycle", // 2
  "car", // 3
  "motorcycle", // 4
  "airplane", // 5
  "bus", // 6
  "train", // 7
  "truck", // 8
  "boat", // 9
  "traffic light", // 10
  "fire hydrant", // 11
  "N/A", // 12 - unused
  "stop sign", // 13
  "parking meter", // 14
  "bench", // 15
  "bird", // 16
  "cat", // 17
  "dog", // 18
  "horse", // 19
  "sheep", // 20
  "cow", // 21
  "elephant", // 22
  "bear", // 23
  "zebra", // 24
  "giraffe", // 25
  "N/A", // 26 - unused
  "backpack", // 27
  "umbrella", // 28
  "N/A", // 29 - unused
  "N/A", // 30 - unused
  "handbag", // 31
  "tie", // 32
  "suitcase", // 33
  "frisbee", // 34
  "skis", // 35
  "snowboard", // 36
  "sports ball", // 37
  "kite", // 38
  "baseball bat", // 39
  "baseball glove", // 40
  "skateboard", // 41
  "surfboard", // 42
  "tennis racket", // 43
  "bottle", // 44
  "N/A", // 45 - unused
  "wine glass", // 46
  "cup", // 47
  "fork", // 48
  "knife", // 49
  "spoon", // 50
  "bowl", // 51
  "banana", // 52
  "apple", // 53
  "sandwich", // 54
  "orange", // 55
  "broccoli", // 56
  "carrot", // 57
  "hot dog", // 58
  "pizza", // 59
  "donut", // 60
  "cake", // 61
  "chair", // 62
  "couch", // 63
  "potted plant", // 64
  "bed", // 65
  "N/A", // 66 - unused
  "dining table", // 67
  "N/A", // 68 - unused
  "N/A", // 69 - unused
  "toilet", // 70
  "N/A", // 71 - unused
  "tv", // 72
  "laptop", // 73
  "mouse", // 74
  "remote", // 75
  "keyboard", // 76
  "cell phone", // 77
  "microwave", // 78
  "oven", // 79
  "toaster", // 80
  "sink", // 81
  "refrigerator", // 82
  "N/A", // 83 - unused
  "book", // 84
  "clock", // 85
  "vase", // 86
  "scissors", // 87
  "teddy bear", // 88
  "hair drier", // 89
  "toothbrush", // 90
];

// The obj2coco head is trained on the 80 "thing" classes (N/A removed). Some
// D-FINE exports emit labels in this dense 0–79 space instead of the sparse
// 0–90 COCO space, so we keep both and resolve labels against whichever fits.
export const COCO_80_CLASSES = COCO_CLASSES.filter((name) => name !== "N/A");

/** Generate a consistent color for a class name (for overlay boxes). */
export function stringToColor(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 80%, 55%)`;
}
