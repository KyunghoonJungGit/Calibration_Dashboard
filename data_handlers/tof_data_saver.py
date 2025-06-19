"""
Save experiment data for Plotly Dash dashboard
"""
import json
import xarray as xr
from pathlib import Path
from datetime import datetime
from typing import List, Any, Dict

def save_tof_experiment_for_dashboard(ds_raw: xr.Dataset, 
                                 qubits: List[Any], 
                                 ds_fit: xr.Dataset,
                                 experiment_type: str = "time_of_flight",
                                 base_dir: str = "D:/Codes/Career/Kyunghoon/Playground/HI_16Jun2025/calibration_dashboard/dashboard_data"):
    """
    실험 데이터를 Dash 대시보드용으로 저장
    
    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw experimental data
    qubits : List[AnyTransmon]
        List of qubit objects
    ds_fit : xr.Dataset
        Fitted parameters
    experiment_type : str
        Type of experiment
    base_dir : str
        Base directory for saving data
    
    Returns
    -------
    Path
        Directory where data was saved
    """
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{experiment_type}_{timestamp}"
    
    # 저장 디렉토리 생성
    save_dir = Path(base_dir) / experiment_id
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Saving experiment data for dashboard ===")
    print(f"Experiment ID: {experiment_id}")
    print(f"Save directory: {save_dir}")
    
    # 1. xarray 데이터셋 저장
    ds_raw_path = save_dir / "ds_raw.nc"
    ds_fit_path = save_dir / "ds_fit.nc"
    
    ds_raw.to_netcdf(ds_raw_path)
    ds_fit.to_netcdf(ds_fit_path)
    print(f"✓ Saved raw data: {ds_raw_path}")
    print(f"✓ Saved fit data: {ds_fit_path}")
    
    # 2. Qubit 정보 추출 및 저장
    qubit_info = tof_exp_extract_qubit_info(qubits, ds_raw)
    qubit_info_path = save_dir / "qubit_info.json"
    
    with open(qubit_info_path, 'w') as f:
        json.dump(qubit_info, f, indent=2)
    print(f"✓ Saved qubit info: {qubit_info_path}")
    
    # 3. 메타데이터 저장
    metadata = {
        "experiment_id": experiment_id,
        "experiment_type": experiment_type,
        "timestamp": timestamp,
        "timestamp_full": datetime.now().isoformat(),
        "data_files": {
            "ds_raw": "ds_raw.nc",
            "ds_fit": "ds_fit.nc",
            "qubit_info": "qubit_info.json"
        },
        "dataset_info": {
            "dimensions": dict(ds_raw.dims),
            "coordinates": list(ds_raw.coords),
            "data_vars": list(ds_raw.data_vars),
            "qubit_count": len(qubits)
        }
    }
    
    metadata_path = save_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")
    
    # 4. 완료 플래그 생성 (파일 감시용)
    complete_flag = save_dir / ".complete"
    complete_flag.touch()
    
    print(f"\n✓ All data saved successfully to: {save_dir}")
    print("=" * 50)
    
    return save_dir


def tof_exp_extract_qubit_info(qubits: List[Any], ds_raw: xr.Dataset) -> Dict:
    """
    Qubit 객체에서 플로팅에 필요한 정보만 추출
    
    Returns
    -------
    Dict
        {
            "grid_locations": ["0,7", "0,6", ...],
            "qubit_names": ["q1", "q2", ...],
            "qubit_mapping": {
                "0,7": {"dataset_index": 0, "qubit_name": "q1"},
                "0,6": {"dataset_index": 1, "qubit_name": "q2"},
                ...
            },
            "dataset_qubit_dim": "qubit"  # ds.qubit.name
        }
    """
    
    grid_locations = []
    qubit_names = []
    qubit_mapping = {}
    
    # 데이터셋의 qubit 차원 이름 확인
    dataset_qubit_dim = ds_raw.qubit.name if hasattr(ds_raw.qubit, 'name') else 'qubit'
    dataset_qubit_values = list(ds_raw.qubit.values)
    
    for idx, q in enumerate(qubits):
        # grid_location 추출
        if hasattr(q, 'grid_location'):
            grid_loc = q.grid_location
        elif hasattr(q, 'name') and ',' in str(q.name):
            grid_loc = q.name
        else:
            # fallback: 인덱스 기반 생성
            grid_loc = f"{idx},0"
        
        grid_locations.append(grid_loc)
        
        # 데이터셋에서의 qubit 이름 확인
        if idx < len(dataset_qubit_values):
            qubit_name = str(dataset_qubit_values[idx])
        else:
            qubit_name = f"q{idx+1}"
        
        qubit_names.append(qubit_name)
        
        # 매핑 정보 저장
        qubit_mapping[grid_loc] = {
            "dataset_index": idx,
            "qubit_name": qubit_name,
            "grid_location": grid_loc
        }
    
    return {
        "grid_locations": grid_locations,
        "qubit_names": qubit_names,
        "qubit_mapping": qubit_mapping,
        "dataset_qubit_dim": dataset_qubit_dim
    }


# 사용 예시: 실험 스크립트의 plot_data 함수에 추가
"""
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    # 기존 matplotlib 플롯
    fig_single_run_fit = plot_single_run_with_fit(
        node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
    )
    fig_averaged_run_fit = plot_averaged_run_with_fit(
        node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
    )
    plt.show()
    
    # Dash 대시보드용 데이터 저장
    use_dash = node.parameters.get('use_dash', True)
    if use_dash:
        save_experiment_for_dashboard(
            node.results["ds_raw"],
            node.namespace["qubits"],
            node.results["ds_fit"],
            experiment_type="time_of_flight"
        )
    
    # Store the generated figures
    node.results["figures"] = {
        "single_run": fig_single_run_fit,
        "averaged_run": fig_averaged_run_fit,
    }
"""