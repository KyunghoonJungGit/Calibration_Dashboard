#!/usr/bin/env python
"""
Calibration Dashboard Server - Main Server Module
Modified to integrate data loading directly into plotters
"""

import os
import sys
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

# 대시보드 관련 임포트
from dash import Dash, dcc, html, Input, Output, State, ALL, MATCH
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

# 로컬 모듈 임포트
from data_handlers.file_watcher import ExperimentDataWatcher
from utils.layout_components import LayoutComponents

from watchdog.observers import Observer


class DashboardServer:
    """모듈화된 캘리브레이션 대시보드 서버 with integrated data loading"""
    
    def __init__(self, port: int = 8091):
        self.port = port
        self.app = None
        self.experiments = {}  # {experiment_id: experiment_dir_path}
        self.experiment_order = []  # 실험 순서
        self.lock = threading.Lock()
        
        # 컴포넌트 초기화
        self.layout_components = LayoutComponents()
        
        # 플로터 레지스트리 (동적 로딩)
        self.plotters = {}
        self._register_plotters()
        
        # 대시보드 초기화
        self.initialize_dashboard()
    
    def _register_plotters(self):
        """사용 가능한 플로터 동적 등록"""
        try:
            # TOF 플로터 등록
            from experiment_plotters.tof_plotter import TOFPlotter
            tof_plotter = TOFPlotter()
            self.plotters[tof_plotter.experiment_type] = tof_plotter
            print(f"✓ Registered plotter: {tof_plotter.experiment_type}")
            
            # Resonator Spec 플로터 등록 (if it has load_experiment_data method)
            try:
                from experiment_plotters.resonator_spec_plotter import ResonatorSpecPlotter
                res_spec_plotter = ResonatorSpecPlotter()
                # Check if it has the new data loading method
                if hasattr(res_spec_plotter, 'load_experiment_data'):
                    self.plotters[res_spec_plotter.experiment_type] = res_spec_plotter
                    print(f"✓ Registered plotter: {res_spec_plotter.experiment_type}")
                else:
                    print(f"⚠️ ResonatorSpecPlotter doesn't have load_experiment_data method yet")
            except Exception as e:
                print(f"⚠️ Could not register ResonatorSpecPlotter: {e}")
            
        except ImportError as e:
            print(f"⚠️ Error importing plotters: {e}")
    
    def _detect_experiment_type(self, experiment_dir: Path) -> Optional[str]:
        """
        Detect experiment type from directory name using keyword matching
        
        Parameters
        ----------
        experiment_dir : Path
            Directory containing experiment files
            
        Returns
        -------
        str or None
            Detected experiment type or None
        """
        dir_name = experiment_dir.name.lower()
        
        # Keyword mapping to experiment types
        # 키워드가 폴더명에 포함되어 있으면 해당 실험 타입으로 인식
        keyword_mapping = {
            'flight': 'time_of_flight',
            'resonator': 'resonator_spectroscopy',
            'qubit_spectroscopy': 'qubit_spectroscopy',
            'ramsey': 'ramsey',
            'rabi': 'rabi_amplitude',
            'power': 'rabi_power',
            't1': 't1',
            't2': 't2',
            'echo': 't2_echo',
            'hello': 'hello_qua'
        }
        
        # Check keywords in order (more specific first)
        for keyword, exp_type in keyword_mapping.items():
            if keyword in dir_name:
                print(f"[DEBUG] Detected '{exp_type}' from keyword '{keyword}' in: {experiment_dir.name}")
                return exp_type
        
        # If not found by keyword, check for data.json
        data_json_path = experiment_dir / 'data.json'
        if data_json_path.exists():
            try:
                with open(data_json_path, 'r') as f:
                    data = json.load(f)
                if 'experiment_type' in data:
                    return data['experiment_type']
                if 'metadata' in data and 'experiment_type' in data['metadata']:
                    return data['metadata']['experiment_type']
            except:
                pass
        
        # Check for metadata.json (old structure)
        metadata_path = experiment_dir / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata.get('experiment_type')
            except:
                pass
        
        # If still not found, return unknown
        print(f"[DEBUG] Could not detect experiment type for: {experiment_dir.name}")
        return 'unknown'
    
    def initialize_dashboard(self):
        """Dash 앱 초기화"""
        self.app = Dash(
            __name__, 
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True  # 동적 컴포넌트를 위해 필요
        )
        
        # 레이아웃 설정
        self.app.layout = self._create_layout()
        
        # 콜백 설정
        self._setup_callbacks()
    
    def _create_layout(self):
        """메인 레이아웃 생성"""
        return dbc.Container([
            # 헤더
            self.layout_components.create_header(),
            
            # 상태 알림 영역
            self.layout_components.create_status_alert(),
            
            # 메인 컨텐츠 영역
            dbc.Row([
                # 왼쪽: 실험 선택
                dbc.Col([
                    self.layout_components.create_experiment_selector_card()
                ], width=3),
                
                # 오른쪽: 디스플레이 옵션 (동적)
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Display Options", className="card-title"),
                            html.Div(
                                id='display-options-content',
                                children=[
                                    html.P("Select an experiment to see display options", 
                                          className="text-muted")
                                ]
                            ),
                            html.Hr(),
                            html.Label("Select Qubits:"),
                            dcc.Dropdown(
                                id='qubit-selector',
                                options=[],
                                value=[],
                                multi=True,
                                placeholder="Select qubits to display"
                            )
                        ])
                    ])
                ], width=9)
            ], className="mb-4"),
            
            # 플롯 영역
            self.layout_components.create_plot_area(),
            
            # 데이터 저장소
            dcc.Store(id='current-experiment-data', storage_type='memory', data=None),
            dcc.Store(id='experiments-store', storage_type='memory', data={}),
            dcc.Store(id='new-experiments-flag', storage_type='memory', data={'has_new': False, 'count': 0}),
            
            # 추가 저장소: 현재 플롯 옵션
            dcc.Store(id='current-plot-options', storage_type='memory', data={}),
            
            # Interval 컴포넌트 (더 빠른 업데이트)
            dcc.Interval(id='check-new-experiments', interval=2000, disabled=False)
        ], fluid=True)
    
    def _setup_callbacks(self):
        """대시보드 콜백 설정"""
        
        # 1. 새 실험 확인 콜백
        @self.app.callback(
            Output('new-experiments-flag', 'data'),
            [Input('check-new-experiments', 'n_intervals')],
            [State('experiments-store', 'data')]
        )
        def check_for_new_experiments(n_intervals, stored_experiments):
            """백그라운드에서 새 실험 확인"""
            if stored_experiments is None:
                stored_experiments = {}
                
            with self.lock:
                new_count = 0
                for exp_id in self.experiment_order:
                    if exp_id not in stored_experiments:
                        new_count += 1
                
                return {'has_new': new_count > 0, 'count': new_count}
        
        # 2. 실험 동기화 콜백
        @self.app.callback(
            [Output('experiments-store', 'data'),
             Output('status-alert', 'children'),
             Output('status-alert', 'is_open'),
             Output('experiment-count', 'children'),
             Output('last-update-time', 'children')],
            [Input('refresh-button', 'n_clicks'),
             Input('new-experiments-flag', 'data')],
            [State('experiments-store', 'data')]
        )
        def sync_experiments(n_clicks, new_flag, stored_experiments):
            """실험 데이터 동기화"""
            import dash
            ctx = dash.callback_context
            
            if stored_experiments is None:
                stored_experiments = {}
            
            print(f"[DEBUG] sync_experiments called - trigger: {ctx.triggered if ctx.triggered else 'initial'}")
            
            if not ctx.triggered:
                trigger = "initial"
            else:
                trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # 새 실험 알림만 표시
            if trigger == 'new-experiments-flag' and new_flag and new_flag.get('has_new'):
                current_count = len(stored_experiments)
                return (
                    dash.no_update,
                    f"🔔 {new_flag['count']} new experiment(s) available! Click Refresh to load.",
                    True,
                    str(current_count),
                    dash.no_update
                )
            
            # 리프레시 버튼을 눌렀을 때만 로드
            if trigger == 'refresh-button' or trigger == 'initial':
                with self.lock:
                    # 실험 데이터 준비
                    experiments_data = {}
                    new_experiments = []
                    
                    for exp_id in self.experiment_order:
                        if exp_id not in stored_experiments:
                            new_experiments.append(exp_id)
                        
                        if exp_id in self.experiments:
                            exp_dir = self.experiments[exp_id]
                            exp_type = self._detect_experiment_type(exp_dir)
                            
                            # Extract timestamp from parent directory (date) and experiment name
                            timestamp = "Unknown"
                            try:
                                # Get parent directory name (date folder)
                                date_folder = exp_dir.parent.name  # e.g., "2025-06-17"
                                exp_name = exp_dir.name  # e.g., "#14_01b_time_of_flight_mw_fem_084413"
                                
                                # Extract time from experiment name (last part)
                                parts = exp_name.split('_')
                                if parts:
                                    time_part = parts[-1]  # e.g., "084413"
                                    # Format: HHMMSS to HH:MM:SS
                                    if len(time_part) == 6 and time_part.isdigit():
                                        formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
                                        timestamp = f"{date_folder} {formatted_time}"
                                    else:
                                        timestamp = f"{date_folder} {time_part}"
                                else:
                                    timestamp = date_folder
                            except:
                                timestamp = "Unknown"
                            
                            experiments_data[exp_id] = {
                                'type': exp_type or 'unknown',
                                'timestamp': timestamp,
                                'exp_dir': str(exp_dir)
                            }
                    
                    print(f"[DEBUG] Prepared experiments_data: {len(experiments_data)} experiments")
                    
                    # 상태 메시지
                    if trigger == 'refresh-button':
                        status_msg = f"✅ Loaded {len(new_experiments)} new experiment(s)" if new_experiments else "✅ Refreshed"
                        show_alert = True
                    else:
                        status_msg = ""
                        show_alert = False
                    
                    update_time = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
                    
                    return (
                        experiments_data,
                        status_msg,
                        show_alert,
                        str(len(self.experiments)),
                        update_time
                    )
            
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        # 3. 실험 목록 업데이트
        @self.app.callback(
            [Output('experiment-selector', 'options'),
             Output('experiment-selector', 'value')],
            [Input('experiments-store', 'data')],
            [State('experiment-selector', 'value')]
        )
        def update_experiment_list(experiments_data, current_value):
            """실험 목록 업데이트"""
            print(f"[DEBUG] update_experiment_list - experiments_data: {experiments_data is not None}")
            
            if not experiments_data:
                return [], None
            
            options = []
            # self.experiment_order를 사용하여 순서 유지
            with self.lock:
                for exp_id in self.experiment_order:
                    if exp_id in experiments_data:
                        exp = experiments_data[exp_id]
                        # 실험 타입에 따른 아이콘 추가
                        icon = self._get_experiment_icon(exp['type'])
                        # 전체 폴더명을 그대로 표시
                        folder_name = exp_id.split('/')[-1] if '/' in exp_id else exp_id
                        date_part = exp_id.split('/')[0] if '/' in exp_id else ''
                        
                        options.append({
                            'label': f"{icon} {folder_name} ({date_part})",
                            'value': exp_id
                        })
            
            print(f"[DEBUG] Generated {len(options)} options from {len(self.experiment_order)} experiments")
            
            # 현재 선택 유지 또는 새로운 선택
            if current_value and current_value in experiments_data:
                return options, current_value
            elif options:
                new_value = options[-1]['value']
                print(f"[DEBUG] Selected new experiment: {new_value}")
                return options, new_value
            
            return options, None
        
        # 3-1. 실험 선택 시 current-experiment-data 업데이트
        @self.app.callback(
            Output('current-experiment-data', 'data'),
            [Input('experiment-selector', 'value')],
            [State('experiments-store', 'data')]
        )
        def update_current_experiment(selected_value, experiments_data):
            """선택된 실험 데이터 업데이트"""
            print(f"[DEBUG] update_current_experiment - selected_value: {selected_value}")
            
            if not selected_value or not experiments_data or selected_value not in experiments_data:
                return None
            
            exp = experiments_data[selected_value]
            
            # Load experiment data using the appropriate plotter
            exp_type = exp['type']
            exp_dir = Path(exp['exp_dir'])
            
            if exp_type in self.plotters:
                plotter = self.plotters[exp_type]
                if hasattr(plotter, 'load_experiment_data'):
                    print(f"[DEBUG] Loading data using {exp_type} plotter")
                    exp_data = plotter.load_experiment_data(exp_dir)
                    if exp_data:
                        # Store loaded data for use in plotting
                        current_data = {
                            'exp_id': selected_value,
                            'type': exp_type,
                            'timestamp': exp['timestamp'],
                            'loaded_data': exp_data
                        }
                        print(f"[DEBUG] Successfully loaded experiment data")
                        return current_data
            
            # Fallback if plotter doesn't support loading
            current_data = {
                'exp_id': selected_value,
                'type': exp['type'],
                'timestamp': exp['timestamp'],
                'exp_dir': exp['exp_dir']
            }
            print(f"[DEBUG] Using basic experiment info (no data loaded)")
            return current_data
        
        # 4. 디스플레이 옵션 동적 업데이트
        @self.app.callback(
            [Output('display-options-content', 'children'),
             Output('current-plot-options', 'data')],
            [Input('current-experiment-data', 'data')],
            [State('current-plot-options', 'data')]
        )
        def update_display_options(current_data, current_options):
            """실험 타입에 따라 디스플레이 옵션 업데이트"""
            print(f"[DEBUG] update_display_options - current_data: {current_data}")
            
            if not current_data:
                return (
                    html.P("Select an experiment to see display options", 
                          className="text-muted"),
                    {}
                )
            
            exp_type = current_data.get('type')
            
            # 해당 실험 타입의 플로터 찾기
            if exp_type in self.plotters:
                plotter = self.plotters[exp_type]
                
                # 플로터에서 디스플레이 옵션 가져오기
                display_components = plotter.get_display_options()
                
                # 기본 옵션 설정
                if current_options is None:
                    current_options = {}
                if exp_type not in current_options:
                    current_options[exp_type] = plotter.get_default_options()
                
                print(f"[DEBUG] Returning display options for {exp_type}")
                return display_components, current_options
            else:
                return (
                    html.P(f"No display options available for {exp_type}", 
                          className="text-muted"),
                    current_options or {}
                )
        
        # 5. 큐빗 선택 옵션 업데이트
        @self.app.callback(
            [Output('qubit-selector', 'options'),
             Output('qubit-selector', 'value'),
             Output('experiment-info', 'children')],
            [Input('current-experiment-data', 'data')]
        )
        def update_qubit_options(current_data):
            """큐빗 선택 옵션 업데이트"""
            print(f"[DEBUG] update_qubit_options - current_data: {current_data}")
            
            if not current_data:
                return [], [], "No experiment selected"
            
            # Check if we have loaded data
            if 'loaded_data' in current_data and current_data['loaded_data']:
                exp_data = current_data['loaded_data']
                qubit_info = exp_data.get('qubit_info', {})
                grid_locations = qubit_info.get('grid_locations', [])
                
                # 큐빗 옵션 생성
                options = [
                    {'label': f"Qubit {loc}", 'value': loc} 
                    for loc in grid_locations
                ]
                
                # 모든 큐빗을 기본으로 선택
                default_value = grid_locations
                
                # 실험 정보 텍스트
                exp_type_display = current_data['type'].replace('_', ' ').title()
                info = (
                    f"📊 Type: {exp_type_display} | "
                    f"🕐 Time: {current_data['timestamp']} | "
                    f"🔢 Qubits: {len(grid_locations)}"
                )
                
                print(f"[DEBUG] Qubit options: {len(options)}, selected: {len(default_value)}")
                
                return options, default_value, info
            else:
                # No loaded data
                return [], [], f"No qubit data available for {current_data.get('type', 'unknown')} experiment"
        
        # 6. TOF 특화 옵션 업데이트 콜백들
        @self.app.callback(
            Output('current-plot-options', 'data', allow_duplicate=True),
            [Input({'type': 'tof-option', 'index': ALL}, 'value')],
            [State('current-experiment-data', 'data'),
             State('current-plot-options', 'data')],
            prevent_initial_call=True
        )
        def update_tof_options(option_values, current_data, current_options):
            """TOF 플롯 옵션 업데이트"""
            if not current_data or current_data.get('type') != 'time_of_flight':
                return current_options or {}
            
            if current_options is None:
                current_options = {}
            
            # 빈 값들이 들어올 수 있으므로 방어적으로 처리
            if not option_values or len(option_values) < 4:
                return current_options
            
            # TOF 옵션 업데이트
            current_options['time_of_flight'] = {
                'plot_type': option_values[0] if option_values[0] else 'averaged',
                'show_options': option_values[1] if option_values[1] else [],
                'max_cols': option_values[2] if option_values[2] else '4',
                'subplot_height': option_values[3] if option_values[3] else 300
            }
            
            print(f"[DEBUG] Updated TOF options: {current_options['time_of_flight']}")
            return current_options
        
        # 7. 메인 플롯 업데이트
        @self.app.callback(
            Output('main-plot', 'figure'),
            [Input('qubit-selector', 'value'),
             Input('current-experiment-data', 'data'),
             Input('current-plot-options', 'data')]
        )
        def update_main_plot(selected_qubits, current_data, stored_options):
            """메인 플롯 업데이트"""
            print(f"[DEBUG] update_main_plot - selected_qubits: {selected_qubits}, current_data: {current_data}")
            
            if not current_data or not selected_qubits:
                return self._create_empty_figure()
            
            exp_type = current_data['type']
            
            # Check if we have loaded data
            if 'loaded_data' not in current_data or not current_data['loaded_data']:
                return self._create_error_figure("No data loaded for this experiment")
            
            exp_data = current_data['loaded_data']
            
            # 플롯 옵션 수집
            plot_options = {}
            if stored_options and exp_type in stored_options:
                plot_options = stored_options[exp_type]
            else:
                # 기본 옵션 사용
                if exp_type in self.plotters:
                    plot_options = self.plotters[exp_type].get_default_options()
            
            print(f"[DEBUG] Plot options: {plot_options}")
            
            # 적절한 플로터로 플롯 생성
            if exp_type in self.plotters:
                try:
                    plotter = self.plotters[exp_type]
                    fig = plotter.create_plot(exp_data, selected_qubits, plot_options)
                    print(f"[DEBUG] Plot created successfully")
                    return fig
                except Exception as e:
                    print(f"[ERROR] Failed to create plot: {e}")
                    import traceback
                    traceback.print_exc()
                    return self._create_error_figure(str(e))
            else:
                return self._create_not_implemented_figure(exp_type)
    
    def _get_experiment_icon(self, exp_type: str) -> str:
        """실험 타입별 아이콘 반환"""
        icons = {
            'time_of_flight': '⏱️',
            'resonator_spectroscopy': '📡',
            'qubit_spectroscopy': '🎯',
            'ramsey': '🌊',
            'rabi_amplitude': '📈',
            'rabi_power': '⚡',
            't1': '⏳',
            't2': '⏰',
            't2_echo': '🔄'
        }
        return icons.get(exp_type, '📊')
    
    def _create_empty_figure(self) -> go.Figure:
        """빈 figure 생성"""
        fig = go.Figure()
        fig.add_annotation(
            text="Select an experiment and qubits to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            template="plotly_white",
            height=600
        )
        return fig
    
    def _create_error_figure(self, error_msg: str) -> go.Figure:
        """에러 figure 생성"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {error_msg}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            template="plotly_white",
            height=600
        )
        return fig
    
    def _create_not_implemented_figure(self, exp_type: str) -> go.Figure:
        """미구현 실험 타입 figure"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Plotter for '{exp_type}' is not yet implemented",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=18, color="orange")
        )
        fig.add_annotation(
            text="Please implement the corresponding plotter class",
            xref="paper", yref="paper",
            x=0.5, y=0.4,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            template="plotly_white",
            height=600
        )
        return fig
    
    def add_experiment_from_directory(self, experiment_dir: Path):
        """디렉토리에서 실험 추가"""
        with self.lock:
            # Create unique experiment ID including date
            try:
                date_folder = experiment_dir.parent.name
                exp_name = experiment_dir.name
                exp_id = f"{date_folder}/{exp_name}"
            except:
                exp_id = experiment_dir.name
            
            if exp_id not in self.experiments:
                self.experiments[exp_id] = experiment_dir
                self.experiment_order.append(exp_id)
                
                # 실험 타입 감지
                exp_type = self._detect_experiment_type(experiment_dir)
                
                print(f"📊 Added to dashboard: {exp_id}")
                print(f"   Type: {exp_type or 'unknown'}")
                print(f"   Total experiments: {len(self.experiments)}")
    
    def scan_existing_experiments(self, base_dir: Path):
        """기존 실험 폴더 스캔 - 날짜별 계층 구조 지원"""
        print(f"\n🔍 Scanning existing experiments in {base_dir}")
        
        experiment_dirs = []
        
        # 날짜 폴더들을 순회
        for date_dir in base_dir.iterdir():
            if date_dir.is_dir() and date_dir.name.startswith('20'):  # YYYY로 시작하는 폴더
                print(f"📅 Scanning date folder: {date_dir.name}")
                
                # 각 날짜 폴더 내의 실험 폴더들을 확인
                for exp_dir in date_dir.iterdir():
                    if exp_dir.is_dir():
                        # .complete 파일이 없어도 일단 추가 (데이터가 있으면)
                        if (exp_dir / 'ds_raw.h5').exists() or (exp_dir / 'data.json').exists():
                            experiment_dirs.append(exp_dir)
                            print(f"  📁 Found experiment: {exp_dir.name}")
        
        # 타임스탬프 기준 정렬
        experiment_dirs.sort(key=lambda x: x.stat().st_mtime)
        
        print(f"\nTotal found: {len(experiment_dirs)} experiments")
        
        # 각 실험 추가
        loaded_count = 0
        
        for exp_dir in experiment_dirs:
            try:
                self.add_experiment_from_directory(exp_dir)
                loaded_count += 1
            except Exception as e:
                print(f"⚠️  Error adding {exp_dir.name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"✅ Successfully added: {loaded_count} experiments")
    
    def run(self, watch_dir: str = "./dashboard_data"):
        """대시보드 서버 실행"""
        watch_path = Path(watch_dir)
        
        if not watch_path.exists():
            watch_path.mkdir(parents=True)
            print(f"📁 Created directory: {watch_path}")
        
        # 기존 실험 스캔
        self.scan_existing_experiments(watch_path)
        
        # 파일 감시 시작 - Modified watcher
        event_handler = ModifiedExperimentDataWatcher(self)
        observer = Observer()
        observer.schedule(event_handler, str(watch_path), recursive=True)
        observer.start()
        
        print("\n" + "="*60)
        print(f"🚀 Dashboard Server is running at http://localhost:{self.port}")
        print(f"📁 Monitoring: {watch_path.absolute()}")
        print(f"📊 Registered plotters: {list(self.plotters.keys())}")
        print("="*60 + "\n")
        
        try:
            # Dash 앱 실행
            self.app.run(
                debug=False,
                port=self.port,
                host='0.0.0.0'
            )
        except KeyboardInterrupt:
            print("\n⏹️  Stopping server...")
            observer.stop()
            observer.join()
            print("✅ Server stopped")


class ModifiedExperimentDataWatcher(ExperimentDataWatcher):
    """Modified file watcher that just tracks directories"""
    
    def __init__(self, dashboard_server):
        """Initialize without data loader"""
        self.dashboard_server = dashboard_server
        self.processing = set()
        self.processed = set()
        self.lock = threading.Lock()
        
        # 설정
        self.completion_marker = '.complete'
        self.processing_delay = 0.5
    
    def on_created(self, event):
        """새 파일/폴더 생성 감지"""
        if event.is_directory:
            return
        
        # 완료 마커 파일 확인
        if event.src_path.endswith(self.completion_marker):
            experiment_dir = Path(event.src_path).parent
            
            with self.lock:
                # 이미 처리 중이거나 처리된 경우 스킵
                if experiment_dir in self.processing or experiment_dir in self.processed:
                    return
                
                self.processing.add(experiment_dir)
            
            # 별도 스레드에서 처리
            thread = threading.Thread(
                target=self._process_experiment_folder,
                args=(experiment_dir,),
                daemon=True
            )
            thread.start()
    
    def _process_experiment_folder(self, experiment_dir: Path):
        """실험 폴더를 처리하여 대시보드에 추가"""
        print(f"\n📁 Processing new experiment: {experiment_dir.name}")
        
        # 파일 쓰기 완료 대기
        import time
        time.sleep(self.processing_delay)
        
        try:
            # Simply add the directory to dashboard
            self.dashboard_server.add_experiment_from_directory(experiment_dir)
            print(f"✅ Successfully added: {experiment_dir.name}")
        except Exception as e:
            print(f"❌ Failed to add {experiment_dir.name}: {e}")
        finally:
            # 처리 완료 후 상태 업데이트
            with self.lock:
                self.processing.discard(experiment_dir)
                self.processed.add(experiment_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calibration Dashboard Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                %(prog)s                   # Run with default settings
                %(prog)s --port 8080       # Run on port 8080
                %(prog)s --watch /data     # Monitor /data directory
                """
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8091, 
        help='Port number (default: 8091)'
    )
    parser.add_argument(
        '--watch', 
        type=str, 
        default='./dashboard_data',
        help='Directory to watch (default: ./dashboard_data)'
    )
    
    args = parser.parse_args()
    
    # 서버 실행
    server = DashboardServer(port=args.port)
    server.run(watch_dir=args.watch)