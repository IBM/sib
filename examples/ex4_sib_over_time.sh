#!/bin/bash

sib_versions=("0.1.8" "0.2.0")

package_name="sib-clustering"

script_base_name = "ex4_sib_over_time"

sleep 10

python "$script_base_name.py" prepare

for version in "${sib_versions[@]}"
do
  echo "$version"
  sed  "s/\$VERSION/${version}/g" "$script_base_name.py" > "$script_base_name_${version}.py"
  pip install $package_name==$version
  sleep 10
  python "$script_base_name_${version}.py" cluster
  rm "$script_base_name_${version}.py"
done

python "$script_base_name.py" evaluate
