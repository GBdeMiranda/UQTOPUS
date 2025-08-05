"""
Template Manager for OpenFOAM Parameter Files

Handles Jinja2 templating for OpenFOAM configuration files.
Simple and focused on the essential functionality.
"""
import os
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pathlib import Path


def process_templates(case_dir, params, template_files):
    """
    Process Jinja2 templates in an OpenFOAM case directory.
    
    Parameters:
        case_dir (str): Path to the case directory
        params (dict): Parameters for template substitution
        template_files (list): List of template files relative to case_dir to process
    """
    env = Environment(
        loader=FileSystemLoader(case_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True
    )
    
    for template_file in template_files:
        template_path = os.path.join(case_dir, template_file)
        
        if not os.path.exists(template_path):
            print(f"Warning: Template file {template_file} not found in {case_dir}")
            continue
            
        try:
            # Load template
            template = env.get_template(template_file)
            
            # Render with parameters
            output = template.render(params)
            
            # Write back to same location
            with open(template_path, 'w') as f:
                f.write(output)
                
        except Exception as e:
            print(f"Error processing template {template_file}: {e}")


def create_template_from_file(input_file, output_file, placeholders):
    """
    Create a Jinja2 template from an existing OpenFOAM file by replacing values with placeholders.
    
    Parameters:
        input_file (str): Path to input OpenFOAM file
        output_file (str): Path to output template file
        placeholders (dict): Dictionary mapping values to placeholder names
                           e.g., {'1e-12': 'permeability', '0.3': 'porosity'}
    """
    try:
        with open(input_file, 'r') as f:
            content = f.read()
        
        # Replace values with Jinja2 placeholders
        for value, placeholder in placeholders.items():
            content = content.replace(str(value), f"{{{{ {placeholder} }}}}")
        
        with open(output_file, 'w') as f:
            f.write(content)
            
        print(f"Template created: {output_file}")
        
    except Exception as e:
        print(f"Error creating template: {e}")


def add_openfoam_filters(env):
    """
    Add OpenFOAM-specific Jinja2 filters.
    
    Parameters:
        env: Jinja2 Environment
    """
    def scientific_notation(value, precision=3):
        """Format number in scientific notation."""
        return f"{float(value):.{precision}e}"
    
    def vector_format(values):
        """Format array as OpenFOAM vector."""
        if hasattr(values, '__iter__'):
            formatted = ' '.join([f"{float(v):.6e}" for v in values])
            return f"({formatted})"
        else:
            return f"({float(values):.6e} 0 0)"
    
    def openfoam_bool(value):
        """Convert boolean to OpenFOAM yes/no."""
        return "yes" if value else "no"
    
    env.filters['scientific'] = scientific_notation
    env.filters['vector'] = vector_format
    env.filters['foam_bool'] = openfoam_bool
    
    return env


def validate_template(template_path, required_params):
    """
    Basic validation of template file.
    
    Parameters:
        template_path (str): Path to template file
        required_params (list): List of required parameter names
        
    Returns:
        bool: True if template is valid
    """
    try:
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Check for required parameters
        missing_params = []
        for param in required_params:
            if f"{{{{{ param }}}}}" not in content and f"{{{{{ param } " not in content:
                missing_params.append(param)
        
        if missing_params:
            print(f"Missing parameters in template {template_path}: {missing_params}")
            return False
        
        # Try to create environment and load template
        env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)))
        template = env.get_template(os.path.basename(template_path))
        
        return True
        
    except Exception as e:
        print(f"Template validation error: {e}")
        return False