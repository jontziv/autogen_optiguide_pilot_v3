#!/usr/bin/env python3
"""
AutoGen Diagnostic Script
=========================
Diagnoses issues with AutoGen + Groq integration and provides specific fixes.
"""

import os
import sys
import traceback
from typing import Dict, List, Any

def check_environment() -> Dict[str, Any]:
    """Check environment setup"""
    print("🔍 Checking Environment...")
    
    results = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "groq_api_key": None,
        "env_file_exists": False,
        "issues": []
    }
    
    # Check Python version
    if sys.version_info < (3, 8):
        results["issues"].append("Python 3.8+ required")
    else:
        print(f"✅ Python {results['python_version']}")
    
    # Check for .env file
    if os.path.exists(".env"):
        results["env_file_exists"] = True
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found")
    
    # Check API key
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and groq_key != "your_groq_api_key_here":
        results["groq_api_key"] = len(groq_key)  # Don't expose actual key
        print(f"✅ GROQ_API_KEY set (length: {len(groq_key)})")
    else:
        results["issues"].append("GROQ_API_KEY not set or using default value")
        print("❌ GROQ_API_KEY not properly set")
    
    return results

def check_autogen_installation() -> Dict[str, Any]:
    """Check AutoGen installation"""
    print("\n🔍 Checking AutoGen Installation...")
    
    results = {
        "core_installed": False,
        "agentchat_installed": False,
        "ext_installed": False,
        "versions": {},
        "issues": []
    }
    
    # Check autogen-core
    try:
        import autogen_core
        results["core_installed"] = True
        results["versions"]["autogen_core"] = getattr(autogen_core, "__version__", "unknown")
        print(f"✅ autogen-core: {results['versions']['autogen_core']}")
    except ImportError as e:
        results["issues"].append(f"autogen-core not installed: {e}")
        print("❌ autogen-core not installed")
    
    # Check autogen-agentchat
    try:
        import autogen_agentchat
        results["agentchat_installed"] = True
        results["versions"]["autogen_agentchat"] = getattr(autogen_agentchat, "__version__", "unknown")
        print(f"✅ autogen-agentchat: {results['versions']['autogen_agentchat']}")
    except ImportError as e:
        results["issues"].append(f"autogen-agentchat not installed: {e}")
        print("❌ autogen-agentchat not installed")
    
    # Check autogen-ext
    try:
        import autogen_ext
        results["ext_installed"] = True
        results["versions"]["autogen_ext"] = getattr(autogen_ext, "__version__", "unknown")
        print(f"✅ autogen-ext: {results['versions']['autogen_ext']}")
    except ImportError as e:
        results["issues"].append(f"autogen-ext not installed: {e}")
        print("❌ autogen-ext not installed")
    
    return results

def test_groq_connection() -> Dict[str, Any]:
    """Test Groq API connection"""
    print("\n🔍 Testing Groq Connection...")
    
    results = {
        "connection_successful": False,
        "model_working": False,
        "error": None
    }
    
    try:
        # Test direct Groq client
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            results["error"] = "No API key available"
            print("❌ No API key for testing")
            return results
        
        client = Groq(api_key=api_key)
        
        # Test simple completion
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Reply with just 'OK'"}],
            max_tokens=10,
            temperature=0
        )
        
        if response.choices and response.choices[0].message.content:
            results["connection_successful"] = True
            results["model_working"] = True
            print("✅ Groq connection working")
        else:
            results["error"] = "Empty response from Groq"
            print("❌ Empty response from Groq")
            
    except Exception as e:
        results["error"] = str(e)
        print(f"❌ Groq connection failed: {e}")
    
    return results

def test_autogen_groq_integration() -> Dict[str, Any]:
    """Test AutoGen + Groq integration"""
    print("\n🔍 Testing AutoGen + Groq Integration...")
    
    results = {
        "client_created": False,
        "agent_created": False,
        "simple_chat": False,
        "error": None
    }
    
    try:
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_agentchat.agents import AssistantAgent
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            results["error"] = "No API key available"
            return results
        
        # Test client creation
        client = OpenAIChatCompletionClient(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "structured_output": True,
                "family": "llama",
            }
        )
        results["client_created"] = True
        print("✅ AutoGen client created")
        
        # Test agent creation
        agent = AssistantAgent(
            "TestAgent",
            model_client=client,
            system_message="You are a test agent. Reply briefly."
        )
        results["agent_created"] = True
        print("✅ AutoGen agent created")
        
        # Test simple interaction
        import asyncio
        from autogen_agentchat.messages import TextMessage
        
        async def test_chat():
            response = await agent.on_messages([TextMessage(content="Say 'WORKING'", source="user")])
            return response
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(test_chat())
            if response and hasattr(response, 'content') and 'WORKING' in response.content.upper():
                results["simple_chat"] = True
                print("✅ AutoGen chat working")
            else:
                print(f"⚠️  Chat response: {response}")
        finally:
            loop.close()
        
    except Exception as e:
        results["error"] = str(e)
        print(f"❌ AutoGen integration failed: {e}")
        traceback.print_exc()
    
    return results

def check_config_issues() -> Dict[str, Any]:
    """Check configuration issues in your codebase"""
    print("\n🔍 Checking Configuration Issues...")
    
    results = {
        "config_loadable": False,
        "orchestrator_creatable": False,
        "issues": []
    }
    
    try:
        from config import safe_config_access
        config = safe_config_access()
        if config:
            results["config_loadable"] = True
            print("✅ Config loads successfully")
            
            # Check if GROQ key is accessible from config
            groq_key = getattr(config, 'groq_api_key', None)
            if groq_key:
                print("✅ Config has groq_api_key")
            else:
                print("⚠️  Config missing groq_api_key")
                results["issues"].append("Config doesn't expose groq_api_key")
        else:
            results["issues"].append("Config is None")
            print("❌ Config failed to load")
            
    except Exception as e:
        results["issues"].append(f"Config import failed: {e}")
        print(f"❌ Config import failed: {e}")
    
    try:
        from autogen_optiguide_system import StreamlitAutoGenOrchestrator
        orchestrator = StreamlitAutoGenOrchestrator()
        if orchestrator.available:
            results["orchestrator_creatable"] = True
            print("✅ AutoGen orchestrator works")
        else:
            print("❌ AutoGen orchestrator not available")
            results["issues"].append("Orchestrator created but not available")
    except Exception as e:
        results["issues"].append(f"Orchestrator creation failed: {e}")
        print(f"❌ Orchestrator creation failed: {e}")
    
    return results

def generate_fix_script(all_results: Dict[str, Any]) -> str:
    """Generate a fix script based on issues found"""
    fix_commands = []
    
    # Environment fixes
    if not all_results["environment"]["groq_api_key"]:
        fix_commands.append("# Set your GROQ API key:")
        fix_commands.append("export GROQ_API_KEY='your_actual_key_here'")
        fix_commands.append("# Or create/update .env file with:")
        fix_commands.append("echo 'GROQ_API_KEY=your_actual_key_here' >> .env")
        fix_commands.append("")
    
    # Installation fixes
    autogen_results = all_results["autogen_installation"]
    if not autogen_results["agentchat_installed"]:
        fix_commands.append("# Install AutoGen packages:")
        fix_commands.append("pip install autogen-agentchat")
        fix_commands.append("")
    
    if not autogen_results["ext_installed"]:
        fix_commands.append("# Install AutoGen extensions:")
        fix_commands.append("pip install autogen-ext[openai]")
        fix_commands.append("")
    
    # Additional fixes based on specific errors
    if all_results.get("groq_connection", {}).get("error"):
        error = all_results["groq_connection"]["error"]
        if "authentication" in error.lower():
            fix_commands.append("# API key authentication failed - check your key:")
            fix_commands.append("# Visit https://console.groq.com/keys to get a valid key")
        elif "rate limit" in error.lower():
            fix_commands.append("# Rate limited - wait a moment and try again")
    
    return "\n".join(fix_commands)

def main():
    """Run comprehensive AutoGen diagnostic"""
    print("🚀 AutoGen + Groq Diagnostic Tool")
    print("=" * 50)
    
    # Run all checks
    results = {
        "environment": check_environment(),
        "autogen_installation": check_autogen_installation(),
        "groq_connection": test_groq_connection(),
        "autogen_integration": test_autogen_groq_integration(),
        "config_issues": check_config_issues()
    }
    
    print("\n" + "=" * 50)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    # Overall status
    all_good = (
        results["environment"]["groq_api_key"] and
        results["autogen_installation"]["agentchat_installed"] and
        results["autogen_installation"]["ext_installed"] and
        results["groq_connection"]["connection_successful"] and
        results["autogen_integration"]["simple_chat"]
    )
    
    if all_good:
        print("Status: ALL SYSTEMS GO!")
        print("Your AutoGen + Groq integration should be working.")
    else:
        print("Status: ISSUES DETECTED")
        print("The following issues need to be resolved:")
        
        # List specific issues
        all_issues = []
        for section, data in results.items():
            if "issues" in data:
                all_issues.extend([f"[{section}] {issue}" for issue in data["issues"]])
            if "error" in data and data["error"]:
                all_issues.append(f"[{section}] {data['error']}")
        
        for issue in all_issues:
            print(f"  - {issue}")
    
    print("\n" + "=" * 50)
    print("RECOMMENDED FIXES")
    print("=" * 50)
    
    fix_script = generate_fix_script(results)
    if fix_script.strip():
        print(fix_script)
    else:
        print("No specific fixes needed - check individual error messages above.")
    
    print("\n" + "=" * 50)
    print("NEXT STEPS")
    print("=" * 50)
    
    if all_good:
        print("1. Run 'streamlit run app.py'")
        print("2. Your AutoGen agents should now work properly")
        print("3. Try asking: 'What if demand for SKU-0001 increases by 50%?'")
    else:
        print("1. Apply the recommended fixes above")
        print("2. Run this diagnostic again to verify fixes")
        print("3. If issues persist, check the detailed error messages")
        print("4. Consider creating a fresh virtual environment")
    
    return results

if __name__ == "__main__":
    main()