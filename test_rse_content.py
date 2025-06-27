#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents'))

try:
    from rse_github_integrator import RSEContent, ContentType
    print("Imports successful")
    
    # Test RSEContent creation
    print("Testing RSEContent creation...")
    content = RSEContent(
        title="Test Content",
        summary="Test summary for validation",
        domain="Software Engineering",
        keywords=["test", "validation"],
        content="This is test content for validation.",
        author="Test Author",
        content_type=ContentType.JSON
    )
    print("RSEContent created successfully")
    
    # Test validation
    print("Testing validation...")
    validation_result = content.validate()
    print(f"Validation result: {validation_result}")
    
    # Test to_dict
    print("Testing to_dict...")
    content_dict = content.to_dict()
    print(f"to_dict successful: {type(content_dict)}")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()