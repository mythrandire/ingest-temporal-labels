import json
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

class IngestTemporalLabels(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="ingest-temporal-labels",
            label="Ingest TemporalDetections",
            description="Ingests a video dataset from a custom temporal detection label format.",
            unlisted=False,
            # Optionally allow delegated (background) execution
            allow_delegated_execution=True,
            allow_immediate_execution=True,
        )
    
    def resolve_input(self, ctx):
        """
        Builds the input form so that the user can configure all 
        `ingest-temporal-labels` parameters before running the operator.
        """
        inputs = types.Object()

        inputs.str(
            "dataset_path",
            label="Dataset Directory",
            description="Path to root of directory containing the video(s). Can be a local path or cloud bucket directory.",
            required=True
        )

        inputs.str(
            "labels_path",
            label="Custom Labels File",
            description="Location of the custom JSON",
            required=True
        )

        inputs.str(
            "dataset_name",
            label="Dataset name",
            description="A name for the resulting dataset",
            required=True
        )

        return types.Property(inputs, view=types.View(label="ingest-temporal-labels parameters"))
    
    def execute(self, ctx):

        params = ctx.params
        temporal_dataset = create_labeled_steps_dataset(
            dataset_directory=params.get('dataset_path'),
            labels_path=params.get('labels_path'),
            dataset_name=params.get('dataset_name'),
        )
        distinct_temporal_dets = temporal_dataset.distinct('procedure_step.detections.label')

        return {"num_temporal_dets": str(distinct_temporal_dets)}
    
    def resolve_output(self, ctx):
        """
        After execution completes, display a read‐only summary to the user.
        """
        outputs = types.Object()
        outputs.int(
            "num_temporal_dets",
            label="Distinct labels found",
            description="List of distinct temporal detection labels found.",
        )
        return types.Property(outputs, view=types.View(label="ingestion summary"))
    
def register(p):
    p.register(IngestTemporalLabels)


def create_labeled_steps_dataset(
        dataset_directory : str,
        labels_path : str,
        dataset_name : str,
        persistent : bool = True,
        overwrite : bool = True
):
    hms_to_seconds = lambda hms: sum(int(x) * 60 ** i 
                                     for i, x in enumerate(
                                            reversed(hms.split(":"))
                                        )
                                    )    
    with open(labels_path, 'r') as f:
        label_data = json.load(f)
    dataset = fo.Dataset(
        name=dataset_name,
        persistent=persistent,
        overwrite=overwrite
    )
    sample = fo.Sample(
        filepath=f"{dataset_directory}/{label_data.get('video_ID')}"
    )
    sample.compute_metadata()
    detections = []
    for step in label_data.get('time_stamp'):
        step_label = step.get('step_label')
        start = hms_to_seconds(step.get('start_time'))
        end = hms_to_seconds(step.get('end_time'))
        detections.append(
            fo.TemporalDetection.from_timestamps(
                [start, end],
                label=step_label,
                sample=sample
            )
        )
        print(f"Step: {step_label} starts at {start} and ends at {end}")
    sample["procedure_step"] = fo.TemporalDetections(detections=detections)
    dataset.add_sample(sample)
    return dataset