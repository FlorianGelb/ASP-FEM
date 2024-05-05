#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <iostream>
#include <fstream>
#include <cmath>
using namespace dealii;


void second_grid() {

    const Point<2> center (0, 0);
    const double inner_radius = 0.4;
    const double outer_radius = 0.6;



    Triangulation<2> corner1;
    Triangulation<2> corner2;
    Triangulation<2> corner3;
    Triangulation<2> corner4;

    Triangulation<2> side1;
    Triangulation<2> side2;
    Triangulation<2> side3;
    Triangulation<2> side4;

    Triangulation<2> total;



    GridGenerator::quarter_hyper_shell(corner1, center, inner_radius, outer_radius);
    GridGenerator::quarter_hyper_shell(corner2, center, inner_radius, outer_radius);
    GridGenerator::quarter_hyper_shell(corner3, center,  inner_radius, outer_radius);
    GridGenerator::quarter_hyper_shell(corner4, center, inner_radius, outer_radius);

    GridGenerator::subdivided_hyper_rectangle(side1, std::vector<unsigned int>{2, 2} ,Point<2>(1.4, -1), Point<2>(1.6, 1));
    GridGenerator::subdivided_hyper_rectangle(side2, std::vector<unsigned int>{2, 2} ,Point<2>(-1.4, -1), Point<2>(-1.6, 1));


    GridGenerator::subdivided_hyper_rectangle(side3, std::vector<unsigned int>{2, 2} ,Point<2>(-1, 1.4), Point<2>(1, 1.6));
    GridGenerator::subdivided_hyper_rectangle(side4, std::vector<unsigned int>{2, 2} ,Point<2>(-1, -1.4), Point<2>(1, -1.6));

    GridTools::rotate(M_PI/2, corner1);
    GridTools::rotate(M_PI, corner2);
    GridTools::rotate(1.5 * M_PI, corner3);
    GridTools::rotate(2 * M_PI, corner4);
    GridTools::shift(Point<2>(-1, 1), corner1);
    GridTools::shift(Point<2>(-1, -1), corner2);
    GridTools::shift(Point<2>(1, -1), corner3);
    GridTools::shift(Point<2>(1, 1), corner4);



    GridGenerator::merge_triangulations(corner1, corner2, total);
    GridGenerator::merge_triangulations(total, corner3, total);
    GridGenerator::merge_triangulations(total, corner4, total);
    GridGenerator::merge_triangulations(total, side1, total);
    GridGenerator::merge_triangulations(total, side2, total);
    GridGenerator::merge_triangulations(total, side3, total);
    GridGenerator::merge_triangulations(total, side4, total);






    std::ofstream out("grid-2.svg");
    GridOut grid_out;
    grid_out.write_svg(total, out);

    std::cout << "Grid written to grid-2.svg" << std::endl;

}



void first_grid()
{


    Triangulation<2> triangulation;
    const Point<2> p1(-1, -1);
    const Point<2> p2(1, 1);
    const Point<2> center(0, 0);
    const double inner_radius = 0.3;
    const double outer_radius = 0.5; // Why not 0.8???
    GridGenerator::hyper_cube(triangulation, -1, 1, true);
    triangulation.refine_global(4);
    for (unsigned int step = 0; step < 3; ++step)
    {
        for (auto &cell : triangulation.active_cell_iterators())
        {
            for (const auto v : cell->vertex_indices())
            {

                const double distance_from_center =
                  center.distance(cell->vertex(v));

                if (std::fabs(distance_from_center - inner_radius) >=
                inner_radius and std::fabs(distance_from_center - outer_radius) <=
                inner_radius )
                {
                    cell->set_refine_flag();
                    break;
                }
            }
        }

        triangulation.execute_coarsening_and_refinement();
    }

    std::ofstream out("grid-1.svg");
    GridOut grid_out;
    grid_out.write_svg(triangulation, out);

    std::cout << "Grid written to grid-1.svg" << std::endl;
}

int main()
{
    //first_grid();
    second_grid();

}
