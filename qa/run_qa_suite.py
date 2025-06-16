#!/usr/bin/env python3
"""
Dr. Orion "TestMaster" Vanguard - Superhuman QA Suite Runner
Main execution script for comprehensive dashboard testing
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from superhuman_qa_orchestrator import SuperhumanQAOrchestrator

def setup_logging():
    """Setup logging configuration"""
    import logging
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("qa/logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_filename = f"qa_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def save_results(results: dict, output_dir: Path):
    """Save QA results to JSON and HTML reports"""
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file = output_dir / f"qa_results_{timestamp}.json"
    
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate HTML report
    html_file = output_dir / f"qa_report_{timestamp}.html"
    generate_html_report(results, html_file)
    
    return json_file, html_file

def generate_html_report(results: dict, output_file: Path):
    """Generate comprehensive HTML report"""
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dr. Orion TestMaster - QA Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header .subtitle {{
            margin: 10px 0 0 0;
            opacity: 0.8;
            font-size: 1.2em;
        }}
        .summary {{
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .passed {{ color: #27ae60; }}
        .failed {{ color: #e74c3c; }}
        .warning {{ color: #f39c12; }}
        .content {{
            padding: 30px;
        }}
        .test-section {{
            margin-bottom: 40px;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }}
        .test-section-header {{
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #e9ecef;
        }}
        .test-section-header h2 {{
            margin: 0;
            color: #2c3e50;
        }}
        .test-section-content {{
            padding: 20px;
        }}
        .test-item {{
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #ddd;
        }}
        .test-item.passed {{
            border-left-color: #27ae60;
            background: #d5f4e6;
        }}
        .test-item.failed {{
            border-left-color: #e74c3c;
            background: #fdeaea;
        }}
        .test-item h4 {{
            margin: 0 0 10px 0;
        }}
        .recommendations {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 10px;
            padding: 20px;
            margin-top: 30px;
        }}
        .recommendations h3 {{
            margin: 0 0 15px 0;
            color: #856404;
        }}
        .recommendations ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin-bottom: 10px;
        }}
        .timestamp {{
            text-align: center;
            padding: 20px;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 Dr. Orion "TestMaster" Vanguard</h1>
            <div class="subtitle">Superhuman Dashboard QA Architect & Inference Maestro</div>
        </div>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Overall Score</h3>
                    <div class="value {'passed' if results.get('overall_score', 0) >= 80 else 'failed'}">
                        {results.get('overall_score', 0):.1f}%
                    </div>
                </div>
                <div class="summary-card">
                    <h3>Test Status</h3>
                    <div class="value {'passed' if results.get('overall_status') == 'passed' else 'failed'}">
                        {results.get('overall_status', 'Unknown').upper()}
                    </div>
                </div>
                <div class="summary-card">
                    <h3>Personas Tested</h3>
                    <div class="value">
                        {len(results.get('persona_testing', {}).get('persona_results', {}))}
                    </div>
                </div>
                <div class="summary-card">
                    <h3>AI Inference Score</h3>
                    <div class="value {'passed' if results.get('ai_inference_validation', {}).get('overall_confidence', 0) >= 0.8 else 'failed'}">
                        {results.get('ai_inference_validation', {}).get('overall_confidence', 0):.1%}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="content">
"""
    
    # Add test sections
    test_sections = [
        ('persona_testing', 'Persona-Based Testing'),
        ('ai_inference_validation', 'AI Inference Validation'),
        ('chaos_engineering', 'Chaos Engineering'),
        ('neural_cache_diagnostics', 'Neural Cache & Prefetch Diagnostics'),
        ('multi_agent_qa_pipeline', 'Multi-Agent QA Pipeline')
    ]
    
    for section_key, section_title in test_sections:
        if section_key in results:
            section_data = results[section_key]
            html_content += f"""
            <div class="test-section">
                <div class="test-section-header">
                    <h2>{section_title}</h2>
                </div>
                <div class="test-section-content">
"""
            
            if section_key == 'persona_testing':
                for persona, persona_data in section_data.get('persona_results', {}).items():
                    status_class = 'passed' if persona_data.get('confidence_score', 0) >= 0.8 else 'failed'
                    html_content += f"""
                    <div class="test-item {status_class}">
                        <h4>{persona}</h4>
                        <p>Confidence Score: {persona_data.get('confidence_score', 0):.1%}</p>
                        <p>Tests Passed: {persona_data.get('tests_passed', 0)}/{persona_data.get('total_tests', 0)}</p>
                    </div>
"""
            
            elif section_key == 'ai_inference_validation':
                for test_name, test_data in section_data.get('test_results', {}).items():
                    status_class = 'passed' if test_data.get('confidence', 0) >= 0.8 else 'failed'
                    html_content += f"""
                    <div class="test-item {status_class}">
                        <h4>{test_name.replace('_', ' ').title()}</h4>
                        <p>Confidence: {test_data.get('confidence', 0):.1%}</p>
                        <p>Status: {test_data.get('status', 'Unknown')}</p>
                    </div>
"""
            
            elif section_key == 'chaos_engineering':
                for experiment_name, experiment_data in section_data.get('experiments', {}).items():
                    status_class = 'passed' if experiment_data.get('status') == 'passed' else 'failed'
                    html_content += f"""
                    <div class="test-item {status_class}">
                        <h4>{experiment_name.replace('_', ' ').title()}</h4>
                        <p>Status: {experiment_data.get('status', 'Unknown')}</p>
                        <p>Recovery Time: {experiment_data.get('recovery_time_seconds', 0)}s</p>
                    </div>
"""
            
            elif section_key == 'neural_cache_diagnostics':
                cache_analysis = section_data.get('cache_analysis', {})
                html_content += f"""
                <div class="test-item passed">
                    <h4>Cache Performance Analysis</h4>
                    <p>Redis Hit Rate: {cache_analysis.get('redis_hit_rate', 0):.1%}</p>
                    <p>Browser Cache Efficiency: {cache_analysis.get('browser_cache_efficiency', 0):.1%}</p>
                    <p>CDN Performance: {cache_analysis.get('cdn_performance_score', 0):.1f}/100</p>
                </div>
"""
            
            elif section_key == 'multi_agent_qa_pipeline':
                pipeline_results = section_data.get('pipeline_results', {})
                for agent_name, agent_data in pipeline_results.items():
                    status_class = 'passed' if agent_data.get('status') == 'passed' else 'failed'
                    html_content += f"""
                    <div class="test-item {status_class}">
                        <h4>{agent_name.replace('_', ' ').title()}</h4>
                        <p>Score: {agent_data.get('score', 0):.1f}%</p>
                        <p>Status: {agent_data.get('status', 'Unknown')}</p>
                    </div>
"""
            
            html_content += """
                </div>
            </div>
"""
    
    # Add recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        html_content += f"""
            <div class="recommendations">
                <h3>🎯 Recommendations</h3>
                <ul>
"""
        for rec in recommendations:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                </ul>
            </div>
"""
    
    html_content += f"""
        </div>
        
        <div class="timestamp">
            Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)

async def run_qa_suite(config_path: str, test_types: list = None, output_dir: str = "qa/reports"):
    """Run the complete QA suite"""
    logger = setup_logging()
    logger.info("🧪 Starting Dr. Orion TestMaster QA Suite")
    
    try:
        # Initialize orchestrator
        orchestrator = SuperhumanQAOrchestrator(config_path)
        await orchestrator.initialize()
        
        # Run selected test types or all tests
        if test_types is None:
            test_types = ['persona', 'inference', 'chaos', 'cache', 'pipeline']
        
        results = {}
        overall_scores = []
        
        # Run persona testing
        if 'persona' in test_types:
            logger.info("🎭 Running persona-based testing...")
            results['persona_testing'] = await orchestrator.run_persona_testing()
            if 'overall_confidence' in results['persona_testing']:
                overall_scores.append(results['persona_testing']['overall_confidence'] * 100)
        
        # Run AI inference validation
        if 'inference' in test_types:
            logger.info("🤖 Running AI inference validation...")
            results['ai_inference_validation'] = await orchestrator.run_ai_inference_validation()
            if 'overall_confidence' in results['ai_inference_validation']:
                overall_scores.append(results['ai_inference_validation']['overall_confidence'] * 100)
        
        # Run chaos engineering
        if 'chaos' in test_types:
            logger.info("💥 Running chaos engineering tests...")
            results['chaos_engineering'] = await orchestrator.run_chaos_engineering()
            chaos_scores = [exp.get('resilience_score', 0) for exp in results['chaos_engineering'].get('experiments', {}).values()]
            if chaos_scores:
                overall_scores.append(sum(chaos_scores) / len(chaos_scores))
        
        # Run neural cache diagnostics
        if 'cache' in test_types:
            logger.info("🧠 Running neural cache diagnostics...")
            results['neural_cache_diagnostics'] = await orchestrator.run_neural_cache_diagnostics()
            cache_score = results['neural_cache_diagnostics'].get('overall_performance_score', 0)
            if cache_score:
                overall_scores.append(cache_score)
        
        # Run multi-agent QA pipeline
        if 'pipeline' in test_types:
            logger.info("🤖 Running multi-agent QA pipeline...")
            results['multi_agent_qa_pipeline'] = await orchestrator._run_multi_agent_qa_pipeline()
            if 'overall_score' in results['multi_agent_qa_pipeline']:
                overall_scores.append(results['multi_agent_qa_pipeline']['overall_score'])
        
        # Calculate overall results
        overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        results['overall_score'] = overall_score
        results['overall_status'] = 'passed' if overall_score >= 80 else 'failed'
        
        # Collect all recommendations
        all_recommendations = []
        for test_result in results.values():
            if isinstance(test_result, dict) and 'recommendations' in test_result:
                all_recommendations.extend(test_result['recommendations'])
        
        results['recommendations'] = all_recommendations
        results['execution_timestamp'] = datetime.now().isoformat()
        
        # Save results
        output_path = Path(output_dir)
        json_file, html_file = save_results(results, output_path)
        
        logger.info(f"✅ QA Suite completed successfully!")
        logger.info(f"📊 Overall Score: {overall_score:.1f}%")
        logger.info(f"📄 JSON Report: {json_file}")
        logger.info(f"🌐 HTML Report: {html_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ QA Suite failed: {str(e)}")
        raise
    finally:
        if 'orchestrator' in locals():
            await orchestrator.cleanup()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Dr. Orion TestMaster - Superhuman QA Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_qa_suite.py --config qa/config.yaml
  python run_qa_suite.py --config qa/config.yaml --tests persona inference
  python run_qa_suite.py --config qa/config.yaml --output-dir custom_reports
"""
    )
    
    parser.add_argument(
        '--config',
        default='qa/config.yaml',
        help='Path to QA configuration file (default: qa/config.yaml)'
    )
    
    parser.add_argument(
        '--tests',
        nargs='*',
        choices=['persona', 'inference', 'chaos', 'cache', 'pipeline'],
        help='Specific test types to run (default: all)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='qa/reports',
        help='Output directory for reports (default: qa/reports)'
    )
    
    args = parser.parse_args()
    
    # Run the QA suite
    try:
        results = asyncio.run(run_qa_suite(
            config_path=args.config,
            test_types=args.tests,
            output_dir=args.output_dir
        ))
        
        # Exit with appropriate code
        exit_code = 0 if results.get('overall_status') == 'passed' else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⚠️  QA Suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ QA Suite failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()