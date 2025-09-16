#!/usr/bin/env bash
set +e

for i in $(seq 1 20); do
  echo "=== Starting episode $i ($(date +'%Y-%m-%d %H:%M:%S')) ==="
  python collect_data_smooth.py --object box --level 1
  status=$?

    if [ $status -ne 0 ]; then
        echo "collect_data.py exited with code $status. Cleaning up…"

        last_folder=$(ls -1dt "$OUT_DIR"/robosuite_automated/teleop_dataset_* 2>/dev/null | head -n1)

        if [ -n "$last_folder" ]; then
            echo "Removing incomplete data in $last_folder"
            rm -rf "$last_folder"
        else
            echo "No folder found to clean up!"
        fi

        echo "Continuing to next object."
        continue
    fi

    echo "Completed for $obj."
done

echo "All done!"

