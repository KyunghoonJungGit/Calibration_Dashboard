#!/usr/bin/env python
"""
Calibration Dashboard Server - Main Server Module
실험 데이터 시각화를 위한 메인 대시보드 서버
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
from HI_16Jun2025.calibration_dashboard.data_handlers.tof_data_loader import ExperimentDataLoader
from utils.layout_components import LayoutComponents
from watchdog.observers import Observer


class DashboardServer:
    """모듈화된 캘리브레이션 대시보드 서버"""
    
    def __init__(self, port: int = 8091):
        self.port = port
        self.app = None
        self.experiments = {}  # {experiment_id: experiment_data}
        self.experiment_order = []  # 실험 순서
        self.lock = threading.Lock()
        
        # 컴포넌트 초기화
        self.layout_components = LayoutComponents()
        self.data_loader = ExperimentDataLoader()
        
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
            
            # 추후 다른 플로터들 추가
            # from experiment_plotters.spectroscopy_plotter import SpectroscopyPlotter
            # from experiment_plotters.rabi_plotter import RabiPlotter
            # from experiment_plotters.ramsey_plotter import RamseyPlotter
            
        except ImportError as e:
            print(f"⚠️ Error importing plotters: {e}")
    
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
            
            # 새 실험 알림만 표시 (자동 로드 제거)
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
            if trigger == 'refresh-button':
                with self.lock:
                    # 실험 데이터 준비
                    experiments_data = {}
                    new_experiments = []
                    
                    for exp_id in self.experiment_order:
                        if exp_id not in stored_experiments:
                            new_experiments.append(exp_id)
                        
                        if exp_id in self.experiments:
                            exp = self.experiments[exp_id]
                            experiments_data[exp_id] = {
                                'type': exp['type'],
                                'timestamp': exp['timestamp'],
                                'grid_locations': exp['qubit_info']['grid_locations'],
                                'qubit_mapping': exp['qubit_info']['qubit_mapping']
                            }
                    
                    print(f"[DEBUG] Prepared experiments_data: {len(experiments_data)} experiments")
                    
                    # 상태 메시지
                    status_msg = f"✅ Loaded {len(new_experiments)} new experiment(s)" if new_experiments else "✅ Refreshed"
                    show_alert = True
                    
                    update_time = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
                    
                    return (
                        experiments_data,
                        status_msg,
                        show_alert,
                        str(len(self.experiments)),
                        update_time
                    )
            
            # 초기 로드
            if trigger == 'initial':
                with self.lock:
                    experiments_data = {}
                    for exp_id in self.experiment_order:
                        if exp_id in self.experiments:
                            exp = self.experiments[exp_id]
                            experiments_data[exp_id] = {
                                'type': exp['type'],
                                'timestamp': exp['timestamp'],
                                'grid_locations': exp['qubit_info']['grid_locations'],
                                'qubit_mapping': exp['qubit_info']['qubit_mapping']
                            }
                    
                    return (
                        experiments_data,
                        "",
                        False,
                        str(len(self.experiments)),
                        f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
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
            print(f"[DEBUG] self.experiment_order: {self.experiment_order}")
            
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
                        options.append({
                            'label': f"{icon} {exp['type']} - {exp['timestamp']}",
                            'value': exp_id
                        })
            
            print(f"[DEBUG] Generated {len(options)} options from {len(self.experiment_order)} experiments")
            
            # 현재 선택 유지 또는 새로운 선택
            if current_value and current_value in experiments_data:
                return options, current_value
            elif options:  # options가 있는지 확인
                new_value = options[-1]['value']  # 마지막 옵션 선택
                print(f"[DEBUG] Selected new experiment: {new_value}")
                return options, new_value
            
            return options, None
        
        # 3-1. 실험 선택 시 current-experiment-data 업데이트 (새로운 콜백)
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
            current_data = {
                'exp_id': selected_value,
                'type': exp['type'],
                'timestamp': exp['timestamp']
            }
            print(f"[DEBUG] Updated current experiment data: {current_data}")
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
            [Input('current-experiment-data', 'data')],
            [State('experiments-store', 'data')]
        )
        def update_qubit_options(current_data, experiments_data):
            """큐빗 선택 옵션 업데이트"""
            print(f"[DEBUG] update_qubit_options - current_data: {current_data}")
            
            if not current_data or not experiments_data:
                return [], [], "No experiment selected"
            
            exp_id = current_data['exp_id']
            if exp_id not in experiments_data:
                return [], [], "Experiment data not found"
            
            exp_data = experiments_data[exp_id]
            
            # 큐빗 옵션 생성
            grid_locations = exp_data['grid_locations']
            options = [
                {'label': f"Qubit {loc}", 'value': loc} 
                for loc in grid_locations
            ]
            
            # 새 실험을 선택했으므로 모든 큐빗을 기본으로 선택
            default_value = grid_locations
            
            # 실험 정보 텍스트
            exp_type_display = exp_data['type'].replace('_', ' ').title()
            info = (
                f"📊 Type: {exp_type_display} | "
                f"🕐 Time: {exp_data['timestamp']} | "
                f"🔢 Qubits: {len(grid_locations)}"
            )
            
            print(f"[DEBUG] Qubit options: {len(options)}, selected: {len(default_value)}")
            
            return options, default_value, info
        
        # 6. TOF 특화 옵션 업데이트 콜백들 추가
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
                'max_cols': option_values[2] if option_values[2] else '2',
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
            
            exp_id = current_data['exp_id']
            exp_type = current_data['type']
            
            with self.lock:
                if exp_id not in self.experiments:
                    print(f"[DEBUG] Experiment {exp_id} not found in self.experiments")
                    return self._create_empty_figure()
                
                exp_data = self.experiments[exp_id]
            
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
            'rabi_power': '⚡'
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
    
    def add_experiment_from_file(self, experiment_data: Dict, experiment_id: str):
        """파일에서 로드한 실험 데이터 추가"""
        with self.lock:
            if experiment_id not in self.experiments:
                self.experiments[experiment_id] = experiment_data
                self.experiment_order.append(experiment_id)
                
                # 실험 타입 통계
                exp_type = experiment_data['type']
                type_count = sum(1 for exp in self.experiments.values() 
                               if exp['type'] == exp_type)
                
                print(f"📊 Added to dashboard: {experiment_id}")
                print(f"   Type: {exp_type} (Total: {type_count})")
                print(f"   Total experiments: {len(self.experiments)}")
    
    def scan_existing_experiments(self, base_dir: Path):
        """기존 실험 폴더 스캔"""
        print(f"\n🔍 Scanning existing experiments in {base_dir}")
        
        experiment_dirs = []
        for item in base_dir.iterdir():
            if item.is_dir() and (item / ".complete").exists():
                experiment_dirs.append(item)
        
        # 타임스탬프 기준 정렬
        experiment_dirs.sort(key=lambda x: x.stat().st_mtime)
        
        print(f"Found {len(experiment_dirs)} existing experiments")
        
        # 각 실험 로드
        loaded_count = 0
        failed_count = 0
        
        for exp_dir in experiment_dirs:
            try:
                # 데이터 로더 사용
                experiment_data = self.data_loader.load_experiment(exp_dir)
                
                if experiment_data:
                    self.add_experiment_from_file(
                        experiment_data,
                        experiment_data['metadata']['experiment_id']
                    )
                    loaded_count += 1
                
            except Exception as e:
                print(f"⚠️  Error loading {exp_dir.name}: {e}")
                failed_count += 1
        
        print(f"✅ Successfully loaded: {loaded_count}")
        if failed_count > 0:
            print(f"❌ Failed to load: {failed_count}")
    
    def run(self, watch_dir: str = "./dashboard_data"):
        """대시보드 서버 실행"""
        watch_path = Path(watch_dir)
        
        if not watch_path.exists():
            watch_path.mkdir(parents=True)
            print(f"📁 Created directory: {watch_path}")
        
        # 기존 실험 스캔
        self.scan_existing_experiments(watch_path)
        
        # 파일 감시 시작
        event_handler = ExperimentDataWatcher(self)
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calibration Dashboard Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run with default settings
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