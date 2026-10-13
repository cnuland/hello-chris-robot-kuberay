from setuptools import setup

setup(
    name="assembly_cell_ros", version="0.6.0", packages=["assembly_cell_ros"],
    data_files=[("share/ament_index/resource_index/packages", ["resource/assembly_cell_ros"]),
                ("share/assembly_cell_ros", ["package.xml"]),
                ("share/assembly_cell_ros/launch", ["launch/assembly_cell.launch.py"])],
    install_requires=["setuptools"], zip_safe=True,
    entry_points={"console_scripts": [
        "cell_sim = assembly_cell_ros.cell_sim_node:main",
        "part_inspector = assembly_cell_ros.part_inspector_node:main",
        "supervisor = assembly_cell_ros.supervisor_node:main",
        "student_controller = assembly_cell_ros.student_controller_node:main",
    ]},
)
