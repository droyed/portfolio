---
title: Prelabel Module
date: 2026-03-03
tags:
  - python
  - yolo
  - data-labeling
  - pre-annotation
  - dataset-cururation
---

[Github repo: `prelabel`](https://github.com/droyed/prelabel)

## Backstory

### Started with Overlapping Masks

I was deep in an instance segmentation project, running YOLO's segmentation model through Ultralytics, when things got messy — literally. Dozens of masks were piling on top of each other, and I couldn't tell which ones were real and which were noise.

The obvious fix was tightening up the IOU threshold for NMS. It cleaned up the visuals, sure, but it also threw out legitimate masks. I needed to *see* what was happening, not just filter it away.

### Finding the Right Debugging Lens

In the age of vibe-coding, my first instinct was to spin up a quick app. But I didn't need "quick" — I needed *advanced*. Bounding boxes layered over segmentation masks, the ability to zoom in, toggle annotations, and actually interrogate the predictions one by one.

After trying several tools, I landed on [Label Studio](https://labelstud.io/). Its interface was exactly the kind of surgical debugging environment I was looking for — and it had room to scale into a full labeling pipeline down the road. Even better, it came with solid API support, which meant I could drive the whole thing from Python.

## Bridging YOLO and Label Studio

The gap was clear: YOLO speaks Python objects, Label Studio speaks API endpoints and a web UI. I didn't want to context-switch between environments or fiddle with manual setup on the Label Studio side. I wanted a workflow that looked like this:

```python
# Predict with YOLO or SAM on image(s)
results = model(images)

# Push predictions straight to Label Studio and visualize
yolo_to_labelstudio(results)

# Setup downstream tasks
```

That's the core idea behind [`prelabel`](https://github.com/droyed/prelabel) — stay in Python, send your results to Label Studio with a single call, and iterate fast.

## Under the Hood

### What Happens When You Call `yolo_to_labelstudio`?

So you've seen the one-liner. But what actually happens when `yolo_to_labelstudio` fires? That single call kicks off a choreographed sequence — walking through your YOLO results, extracting predictions, connecting to Label Studio, optionally creating a project from scratch, and batch-importing everything. At the heart of it sits a class called `LabelStudioClient`. Think of it as a remote control for Label Studio — you write a few lines of Python, and it handles the API calls behind the scenes.

A few creature comforts came along with the core plumbing: per-class color palettes (or an auto-generated maximally distinct scheme so you're not squinting at ten shades of blue), built-in confidence-score filtering, and support for bounding boxes and segmentation masks, with more annotation types on the way.

The rest of this post walks through what that remote control can do, and why you might want to pick it up directly.

### Connecting to Label Studio

Before anything else, you need a running Label Studio instance and an API key. Once you have both, creating a client is straightforward:

```python
from prelabel import LabelStudioClient

client = LabelStudioClient(port=8080, api_key="your-api-key-here")
```

The `port` is whatever port Label Studio is running on (commonly `8080`), and the `api_key` is the token you can find in your Label Studio account settings. Once this line runs without errors, you're connected and ready to go.

When you call `yolo_to_labelstudio`, it creates this client internally using your port and an API token pulled from an environment variable. But nothing stops you from creating one yourself and calling the methods directly — which is exactly what the rest of this walkthrough covers.

### Creating Projects: Picking the Right Annotation Type

Label Studio organizes everything into **projects**. Each project has a labeling configuration — what kind of annotations you want — and a set of tasks (images or data to annotate). `LabelStudioClient` gives you several convenient ways to create projects depending on your annotation type.

If you're working on **object detection**, where you need to draw rectangles around objects:

```python
project_id = client.create_bbox_project(
    title="Street Signs Detection",
    labels=["Stop Sign", "Speed Limit", "Yield"]
)
```

Need more precise outlines? **Polygons** let you trace around irregular shapes:

```python
project_id = client.create_polygon_project(
    title="Building Footprints",
    labels=["Residential", "Commercial", "Industrial"]
)
```

For **pixel-level segmentation** tasks where you paint masks over regions:

```python
project_id = client.create_brush_project(
    title="Land Cover Classification",
    labels=["Forest", "Water", "Urban", "Agriculture"]
)
```

And if you want full control — like specifying label colors — the generic method has you covered:

```python
labels = [
    {"name": "Person", "color": "#FFA39E"},
    {"name": "Car", "color": "#D4380D"},
    {"name": "Bicycle", "color": "#1890FF"}
]

project_id = client.create_cv_project_generic(
    title="Traffic Scene Segmentation",
    labels=labels,
    label_type="BrushLabels"  # or "RectangleLabels"
)
```

This is the same method that `yolo_to_labelstudio` calls when you don't pass a `projectID` — it figures out whether to use `BrushLabels` or `RectangleLabels` based on your `task_type`, gathers the class names from your YOLO results, and creates the project for you. But when you call `create_cv_project_generic` directly, you're in full control of the label names, colors, and type.

### Uploading Images

Once your project is created, you need to feed it data. The `import_local_images` method uploads all valid images (`.jpg`, `.jpeg`, `.png`, `.bmp`) from a folder on your machine:

```python
count = client.import_local_images(
    project_id=project_id,
    image_directory="/path/to/your/images"
)
print(f"Uploaded {count} images to the project.")
```

It loops through every image file in the directory and pushes them to your project. No need to upload one by one through the browser.

> **Note:** For this method to work, your Label Studio instance needs to have `LOCAL_FILES_SERVING_ENABLED=true` in its environment configuration.

### Importing Pre-Annotated Data: The Real Power Move

This is arguably the most powerful feature of the class — and the one that `yolo_to_labelstudio` leans on most heavily. Instead of starting from scratch, you can push **model predictions** into Label Studio so that annotators can review, correct, and approve them rather than draw everything from zero.

When you have dozens or hundreds of images to push, the batch method is what you want:

```python
batch_data = [
    {"image_path": "/data/img_001.jpg", "predictions": predictions_001},
    {"image_path": "/data/img_002.jpg", "predictions": predictions_002},
    {"image_path": "/data/img_003.jpg", "predictions": predictions_003},
    # ... as many as you need
]

total = client.import_preannotated_tasks_batch(
    project_id=project_id,
    batch_data=batch_data,
    model_version="yolov8-custom",
    batch_size=25  # sends 25 images per API call
)
print(f"Imported {total} tasks in total.")
```

The batch method is smart about chunking — it breaks your data into groups of `batch_size` to avoid overwhelming the server with massive payloads. It also gracefully skips any images it can't find, printing a warning instead of crashing.

Inside `yolo_to_labelstudio`, this is exactly what happens after your YOLO results are processed. Each result's image path and extracted predictions are bundled into that `batch_data` list, and `import_preannotated_tasks_batch` takes it from there. The `batch_size`, `model_version`, and `conf_threshold` parameters you pass to `yolo_to_labelstudio` flow straight through to this step.

### Monitoring Your Projects

As your projects grow, you'll want a quick way to see what's happening across all of them.

```python
client.list_projects_summary()
```

This prints a neat table to your console showing every project's ID, title, number of classes, task count, annotation progress, and creation date:

```
==========================================================================================
ID   | Title                  | Classes | Tasks  | Annotated | Progress | Annots | Created Date
------------------------------------------------------------------------------------------
1    | Street Signs Detection | 3       | 150    | 45        | 30.0%    | 45     | 2025-01-15
2    | Land Cover             | 4       | 500    | 500       | 100.0%   | 620    | 2025-02-01
==========================================================================================
```

If you need the same information in code — say, to log it or display it in a dashboard — there's a dictionary version:

```python
summaries = client.get_projects_summary()

for project in summaries:
    print(f"{project['Title']}: {project['Progress']} complete")
```

### Checking if a Project Exists

Before pushing data to a project, it's good practice to verify it exists. The client handles this internally (and `yolo_to_labelstudio` calls `project_exists` with `raise_on_missing=True` whenever you pass a `projectID`), but you can also check explicitly:

```python
# Raises an error if the project doesn't exist
client.project_exists(project_id=42)

# Or check silently without raising
exists = client.project_exists(project_id=42, raise_on_missing=False)
if exists:
    print("Project found!")
else:
    print("Project not found.")
```

### Exporting Annotations

After your team finishes labeling, you'll want to pull those annotations out in a format your training pipeline understands. The client offers purpose-built export methods for the most common formats.

YOLO format for bounding boxes:

```python
client.export_bbox_yolo(project_id=1, output_path="yolo_bboxes.zip")
```

COCO format for polygons:

```python
client.export_polygon_coco(project_id=2, output_path="coco_polygons.zip")
```

PNG masks for brush segmentation:

```python
client.export_brush_png(project_id=3, output_path="png_masks.zip")
```

Each of these downloads a ZIP file to the path you specify, containing the annotations in the respective format. From there, you can feed them directly into your training scripts. And if you need a different format, the underlying `export_annotations` method is also available:

```python
client.export_annotations(project_id=1, export_type="JSON", output_path="annotations.zip")
```

### Housekeeping: Cleanup and Deletion

Over time, you might accumulate test projects or empty projects that clutter your instance. The client has a couple of methods to keep things tidy.

To remove all projects with zero tasks:

```python
deleted = client.cleanup_empty_projects()
print(f"Removed projects: {deleted}")
```

And the nuclear option — removing every project on the instance, including their tasks and annotations:

```python
count = client.delete_all_projects()
print(f"Deleted {count} projects.")
```

Use that last one with caution. It's handy for resetting a development environment, but definitely not something to run in production.

## The Full Picture

Let's zoom out and see how all these pieces fit together in practice. Here's what a realistic end-to-end workflow looks like — starting from YOLO predictions, flowing through `yolo_to_labelstudio`, and extending into the full labeling lifecycle using `LabelStudioClient` directly:

```python
from ultralytics import YOLO
from prelabel import yolo_to_labelstudio, LabelStudioClient

# --- Step 1: Predict ---
model = YOLO("yolov8n-seg.pt")
results = model("path/to/images/")

# --- Step 2: Push predictions to Label Studio ---
project_id = yolo_to_labelstudio(
    results,
    task_type="segmentation",
    project_title="Segmentation Review",
    model_version="yolov8n-seg-v1",
    conf_threshold=0.3
)

# --- Step 3: Monitor progress ---
client = LabelStudioClient(port=8080, api_key="your-key")
client.list_projects_summary()

# --- Step 4: Export when labeling is done ---
client.export_brush_png(project_id, "reviewed_masks.zip")
```

Four steps — from model inference to exported, human-reviewed labels — all from a Python script. No clicking through menus, no copy-pasting URLs, no manual file uploads.

`yolo_to_labelstudio` handles the heavy lifting of translating YOLO's world into Label Studio's world. And when you need finer control — creating projects with custom colors, uploading raw images, monitoring progress, exporting in specific formats, cleaning up old projects — `LabelStudioClient` is right there, one import away.

## Why Would You Reach for `prelabel`?

You've trained a model. The metrics look reasonable, but when you run inference on fresh images, something feels off — masks bleed into each other, bounding boxes are too tight, a "car" keeps getting tagged as "truck." The numbers say the model is fine. Your eyes say otherwise. That's the gap `prelabel` lives in.

You could write a matplotlib loop, but it falls apart the moment you need to zoom, toggle annotations, or inspect a mask at pixel level. You could set up Label Studio manually, but the moment you're iterating on a model, that manual setup quietly eats hours from your week. `prelabel` collapses both problems into one function call — and you're staring at your model's output in a professional annotation interface, ready to interrogate every prediction.

The deeper payoff is what comes *after*. Once predictions are in Label Studio, you're one step away from a full correction workflow: fix, relabel, export, retrain. Everyone knows this loop in theory — it just doesn't happen often enough because the plumbing between steps is tedious. `prelabel` makes that plumbing disappear, so each spin of the cycle makes your model a little bit better.

The whole thing started because I couldn't tell which overlapping masks were real. If you've ever wished you could just *click* on a prediction and see what's going on — that's exactly the itch [`prelabel`](https://github.com/droyed/prelabel) scratches.

