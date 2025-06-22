"""
Time of Flight Plotter Module
Time of Flight 실험 데이터 시각화를 위한 플로터
Integrated data loading following matplotlib code structure
"""
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import xarray as xr
import json
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc

from .base_plotter import ExperimentPlotter


class TOFPlotter(ExperimentPlotter):
    """Time of Flight 실험 플로터 with integrated data loading"""
    
    def __init__(self):
        """TOF 플로터 초기화"""
        super().__init__()
        
        # TOF 특화 설정
        self.adc_range = (-0.5, 0.5)  # ADC 범위 (mV)
        self.adc_full_range = (-0.6, 0.6)  # ADC 전체 표시 범위 (mV)
        
        # 색상 설정
        self.trace_colors = {
            'I': 'blue',
            'Q': 'red',
            'I_single': 'lightblue',
            'Q_single': 'lightcoral'
        }
    
    @property
    def experiment_type(self) -> str:
        """실험 타입"""
        return "time_of_flight"
    
    def load_experiment_data(self, experiment_dir: Path) -> Optional[Dict]:
        """
        Load TOF experiment data from directory
        Following the matplotlib code structure
        
        Parameters
        ----------
        experiment_dir : Path
            Directory containing experiment data files
            
        Returns
        -------
        Dict or None
            Loaded experiment data or None if failed
        """
        try:
            print(f"\n=== Loading TOF data from {experiment_dir} ===")
            
            # File paths
            ds_raw_path = experiment_dir / "ds_raw.h5"
            ds_fit_path = experiment_dir / "ds_fit.h5"
            data_json_path = experiment_dir / "data.json"
            node_json_path = experiment_dir / "node.json"
            
            # Check if all required files exist
            required_files = [ds_raw_path, ds_fit_path, data_json_path, node_json_path]
            missing_files = [f for f in required_files if not f.exists()]
            if missing_files:
                raise FileNotFoundError(f"Missing required files: {[f.name for f in missing_files]}")
            
            # Load xarray datasets
            ds_raw = xr.open_dataset(ds_raw_path)
            ds_fit = xr.open_dataset(ds_fit_path)
            
            # Load JSON files
            with open(data_json_path, 'r') as f:
                data_json = json.load(f)
            
            with open(node_json_path, 'r') as f:
                node_json = json.load(f)
            
            # Extract qubit information
            qubits = ds_raw['qubit'].values
            n_qubits = len(qubits)
            print(f"Number of qubits: {n_qubits}")
            print(f"Qubits: {qubits}")
            
            # Extract fitting success information
            success = ds_fit['success'].values
            print(f"Fitting success for each qubit: {success}")
            
            # Create qubit info structure compatible with dashboard
            grid_locations = []
            qubit_names = []
            qubit_mapping = {}
            
            for idx, qubit in enumerate(qubits):
                # Extract grid location from qubit name (e.g., "q0-7" -> "0,7")
                if isinstance(qubit, str) and '-' in qubit:
                    parts = qubit.replace('q', '').split('-')
                    grid_loc = f"{parts[0]},{parts[1]}"
                else:
                    # Fallback
                    grid_loc = f"0,{idx}"
                
                grid_locations.append(grid_loc)
                qubit_names.append(str(qubit))
                
                qubit_mapping[grid_loc] = {
                    "dataset_index": idx,
                    "qubit_name": str(qubit),
                    "grid_location": grid_loc
                }
            
            qubit_info = {
                "grid_locations": grid_locations,
                "qubit_names": qubit_names,
                "qubit_mapping": qubit_mapping,
                "dataset_qubit_dim": "qubit"
            }
            
            # Extract metadata
            metadata = {
                "experiment_id": experiment_dir.name,
                "experiment_type": "time_of_flight"
            }
            
            # Add timestamp if available in data.json
            if 'timestamp' in data_json:
                metadata['timestamp'] = data_json['timestamp']
            elif 'metadata' in data_json and 'timestamp' in data_json['metadata']:
                metadata['timestamp'] = data_json['metadata']['timestamp']
            else:
                metadata['timestamp'] = "Unknown"
            
            # Compile experiment data
            experiment_data = {
                'type': 'time_of_flight',
                'ds_raw': ds_raw,
                'ds_fit': ds_fit,
                'data_json': data_json,
                'node_json': node_json,
                'qubit_info': qubit_info,
                'metadata': metadata,
                'timestamp': metadata['timestamp']
            }
            
            print(f"✓ Successfully loaded TOF data")
            return experiment_data
            
        except Exception as e:
            print(f"❌ Error loading TOF experiment from {experiment_dir}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_display_options(self) -> List[Any]:
        """TOF 전용 디스플레이 옵션"""
        return [
            # 플롯 타입 선택
            html.Div([
                html.Label("Plot Type:", className="font-weight-bold mb-2"),
                dcc.RadioItems(
                    id={'type': 'tof-option', 'index': 0},
                    options=[
                        {"label": "Averaged Run", "value": "averaged"},
                        {"label": "Single Run", "value": "single"},
                        {"label": "Both", "value": "both"}
                    ],
                    value="averaged",
                    className="mb-3",
                    labelStyle={'display': 'block', 'margin-bottom': '5px'}
                )
            ]),
            
            html.Hr(),
            
            # 추가 옵션들
            html.Div([
                html.Label("Display Options:", className="font-weight-bold mb-2"),
                dcc.Checklist(
                    id={'type': 'tof-option', 'index': 1},
                    options=[
                        {"label": "Show TOF Line", "value": "show_tof"},
                        {"label": "Show ADC Range", "value": "show_adc"},
                        {"label": "Auto-scale Y-axis", "value": "auto_scale"}
                    ],
                    value=["show_tof", "show_adc"],
                    className="mb-3",
                    labelStyle={'display': 'block', 'margin-bottom': '5px'}
                )
            ]),
            
            html.Hr(),
            
            # 레이아웃 옵션
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Max Columns:", className="mb-2"),
                        dcc.Dropdown(
                            id={'type': 'tof-option', 'index': 2},
                            options=[
                                {"label": "1", "value": "1"},
                                {"label": "2", "value": "2"},
                                {"label": "3", "value": "3"},
                                {"label": "4", "value": "4"}
                            ],
                            value="4",
                            style={"width": "100%"}
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Height per subplot:", className="mb-2"),
                        dcc.Input(
                            id={'type': 'tof-option', 'index': 3},
                            type="number",
                            value=300,
                            min=200,
                            max=600,
                            step=50,
                            style={"width": "100%"}
                        )
                    ], width=6)
                ])
            ])
        ]
    
    def get_default_options(self) -> Dict:
        """기본 옵션"""
        return {
            'plot_type': 'averaged',
            'show_options': ['show_tof', 'show_adc'],
            'max_cols': '4',
            'subplot_height': 300
        }
    
    def create_plot(self, exp_data: Dict, selected_qubits: List[str], 
                   plot_options: Dict) -> go.Figure:
        """TOF 플롯 생성"""
        # 옵션 추출
        plot_type = plot_options.get('plot_type', 'averaged')
        show_options = plot_options.get('show_options', ['show_tof', 'show_adc'])
        show_tof = 'show_tof' in show_options
        show_adc = 'show_adc' in show_options
        auto_scale = 'auto_scale' in show_options
        
        # 동적 레이아웃 설정
        max_cols = int(plot_options.get('max_cols', '4'))
        subplot_height = int(plot_options.get('subplot_height', 300))
        
        # 원래 설정 임시 변경
        original_max_cols = self.MAX_COLS
        original_subplot_height = self.SUBPLOT_HEIGHT
        self.MAX_COLS = max_cols
        self.SUBPLOT_HEIGHT = subplot_height
        
        try:
            # 데이터 추출
            ds_raw = exp_data['ds_raw']
            ds_fit = exp_data['ds_fit']
            qubit_info = exp_data['qubit_info']
            
            n_qubits = len(selected_qubits)
            
            # 레이아웃 계산
            rows_multiplier = 2 if plot_type == 'both' else 1
            rows, cols, total_height = self.get_subplot_layout(n_qubits, rows_multiplier)
            
            # 서브플롯 타이틀 생성
            subplot_titles = self._generate_subplot_titles(selected_qubits, plot_type)
            
            # Figure 생성
            fig = self.create_subplots_figure(rows, cols, subplot_titles, total_height)
            
            # 플롯 데이터 추가
            self._add_tof_data(
                fig, ds_raw, ds_fit, qubit_info,
                selected_qubits, plot_type, rows, cols,
                show_tof, show_adc, auto_scale
            )
            
            # 축 레이블 업데이트
            fig.update_xaxes(title_text="Time [ns]")
            fig.update_yaxes(title_text="Readout amplitude [mV]")
            
            # 서브플롯 타이틀 위치 조정
            self.update_subplot_titles_position(fig, subplot_titles)
            
            # 공통 레이아웃 적용
            title = f"Time of Flight - {plot_type.capitalize()} Run"
            self.apply_common_layout(fig, title)
            
            return fig
            
        finally:
            # 원래 설정 복원
            self.MAX_COLS = original_max_cols
            self.SUBPLOT_HEIGHT = original_subplot_height
    
    def _generate_subplot_titles(self, selected_qubits: List[str], 
                               plot_type: str) -> List[str]:
        """서브플롯 타이틀 생성"""
        if plot_type == 'both':
            titles = []
            for loc in selected_qubits:
                titles.append(f"{loc} - Averaged")
            for loc in selected_qubits:
                titles.append(f"{loc} - Single Run")
        else:
            titles = selected_qubits
        
        return titles
    
    def _add_tof_data(self, fig: go.Figure, ds_raw: Any, ds_fit: Any,
                     qubit_info: Dict, selected_qubits: List[str],
                     plot_type: str, rows: int, cols: int,
                     show_tof: bool, show_adc: bool, auto_scale: bool):
        """TOF 데이터를 figure에 추가"""
        # 범례 표시 여부 추적
        show_legends = {
            'I': True,
            'Q': True,
            'I_single': True,
            'Q_single': True
        }
        
        for idx, grid_location in enumerate(selected_qubits):
            col = (idx % cols) + 1
            
            try:
                # Get qubit name from grid location
                qubit_name = qubit_info['qubit_mapping'][grid_location]['qubit_name']
                
                # Get data for this qubit
                qubit_data = ds_raw.sel(qubit=qubit_name)
                
                # 시간 축
                time = qubit_data.readout_time.values
                
                # TOF delay 값 가져오기
                delay_value = self._get_delay_value(ds_fit, qubit_name)
                
                # Averaged plot
                if plot_type in ['averaged', 'both']:
                    row = (idx // cols) + 1
                    self._add_averaged_plot(
                        fig, qubit_data, time, delay_value,
                        row, col, show_legends, show_tof, show_adc, auto_scale
                    )
                
                # Single run plot
                if plot_type in ['single', 'both']:
                    if plot_type == 'both':
                        row = rows // 2 + (idx // cols) + 1
                    else:
                        row = (idx // cols) + 1
                    
                    self._add_single_run_plot(
                        fig, qubit_data, time, delay_value,
                        row, col, show_legends, plot_type,
                        show_tof, show_adc, auto_scale
                    )
                    
            except Exception as e:
                print(f"Error plotting {grid_location}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def _get_delay_value(self, ds_fit: Any, qubit_name: str) -> float:
        """TOF delay 값 추출"""
        try:
            return float(ds_fit.sel(qubit=qubit_name).delay.values)
        except:
            try:
                return float(ds_fit.delay.values)
            except:
                print(f"Warning: Could not extract delay value for {qubit_name}")
                return 0.0
    
    def _add_averaged_plot(self, fig: go.Figure, qubit_data: Any, time: Any,
                          delay_value: float, row: int, col: int,
                          show_legends: Dict, show_tof: bool, 
                          show_adc: bool, auto_scale: bool):
        """Averaged 데이터 플롯 추가"""
        # Convert to mV
        adcI_mV = qubit_data.adcI.values * 1e3
        adcQ_mV = qubit_data.adcQ.values * 1e3
        
        # I quadrature
        fig.add_trace(
            go.Scatter(
                x=time,
                y=adcI_mV,
                name='I' if show_legends['I'] else None,
                line=dict(color=self.trace_colors['I'], width=2),
                showlegend=show_legends['I'],
                legendgroup='I'
            ),
            row=row, col=col
        )
        show_legends['I'] = False
        
        # Q quadrature
        fig.add_trace(
            go.Scatter(
                x=time,
                y=adcQ_mV,
                name='Q' if show_legends['Q'] else None,
                line=dict(color=self.trace_colors['Q'], width=2),
                showlegend=show_legends['Q'],
                legendgroup='Q'
            ),
            row=row, col=col
        )
        show_legends['Q'] = False
        
        # TOF line
        if show_tof:
            fig.add_vline(
                x=delay_value,
                line_dash="dash",
                line_color="black",
                annotation_text="TOF" if row == 1 and col == 1 else None,
                row=row, col=col
            )
        
        # ADC Range
        if show_adc:
            fig.add_shape(
                type="rect",
                x0=time[0], x1=time[-1],
                y0=self.adc_range[0], y1=self.adc_range[1],
                fillcolor="gray",
                opacity=0.2,
                layer="below",
                line_width=0,
                row=row, col=col
            )
            
            # Add ADC Range text
            fig.add_annotation(
                text="ADC Range",
                x=time[0] + (time[-1] - time[0]) * 0.02,
                y=self.adc_range[1] * 0.9,
                xref=f"x{col}" if row == 1 else f"x{col + (row-1)*cols}",
                yref=f"y{col}" if row == 1 else f"y{col + (row-1)*cols}",
                showarrow=False,
                font=dict(size=8, color="gray")
            )
        
        # Y축 범위 설정
        if auto_scale:
            y_data = np.concatenate([adcI_mV, adcQ_mV])
            y_min, y_max = self.calculate_y_range(y_data)
        else:
            y_min, y_max = self.adc_full_range
        
        fig.update_yaxes(range=[y_min, y_max], row=row, col=col)
        fig.update_xaxes(range=[0, 1000], row=row, col=col)
    
    def _add_single_run_plot(self, fig: go.Figure, qubit_data: Any, time: Any,
                           delay_value: float, row: int, col: int,
                           show_legends: Dict, plot_type: str,
                           show_tof: bool, show_adc: bool, auto_scale: bool):
        """Single run 데이터 플롯 추가"""
        # Single run 데이터 (mV로 변환)
        single_i = qubit_data.adc_single_runI.values * 1e3
        single_q = qubit_data.adc_single_runQ.values * 1e3
        
        # I trace
        show_i_legend = show_legends['I_single'] and plot_type == 'single'
        fig.add_trace(
            go.Scatter(
                x=time,
                y=single_i,
                name='I (single)' if show_i_legend else None,
                line=dict(color=self.trace_colors['I_single'], width=1),
                showlegend=show_i_legend,
                legendgroup='I_single'
            ),
            row=row, col=col
        )
        if plot_type == 'single':
            show_legends['I_single'] = False
        
        # Q trace
        show_q_legend = show_legends['Q_single'] and plot_type == 'single'
        fig.add_trace(
            go.Scatter(
                x=time,
                y=single_q,
                name='Q (single)' if show_q_legend else None,
                line=dict(color=self.trace_colors['Q_single'], width=1),
                showlegend=show_q_legend,
                legendgroup='Q_single'
            ),
            row=row, col=col
        )
        if plot_type == 'single':
            show_legends['Q_single'] = False
        
        # TOF line
        if show_tof:
            fig.add_vline(
                x=delay_value,
                line_dash="dash",
                line_color="black",
                row=row, col=col
            )
        
        # ADC Range
        if show_adc:
            fig.add_shape(
                type="rect",
                x0=time[0], x1=time[-1],
                y0=self.adc_range[0] * 6, y1=self.adc_range[1] * 6,  # Larger range for single run
                fillcolor="gray",
                opacity=0.2,
                layer="below",
                line_width=0,
                row=row, col=col
            )
        
        # Y축 범위 설정
        if auto_scale:
            y_data_single = np.concatenate([single_i, single_q])
            y_min_single, y_max_single = np.min(y_data_single), np.max(y_data_single)
            
            # 신호가 작으면 데이터에 맞춤
            if y_max_single < 0.1 and y_min_single > -0.1:
                y_min, y_max = self.calculate_y_range(y_data_single)
            else:
                # 신호가 크면 고정 범위
                y_min, y_max = -3, 3
        else:
            y_min, y_max = -3, 3  # Larger range for single run data
        
        fig.update_yaxes(range=[y_min, y_max], row=row, col=col)
        fig.update_xaxes(range=[0, 1000], row=row, col=col)