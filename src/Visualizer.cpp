#include "Visualizer.h"

Visualizer::Visualizer() {}

void Visualizer::drawDelaunay(const cv::Mat &current, std::vector<cv::Point2f> &good_new, const cv::Scalar& color)
{
    if (good_new.size() < 3) // devo avere almeno 3 punti
        return;

    cv::Size size = current.size(); //trovo le dimensioni del frame 
    cv::Rect rect(0, 0, size.width, size.height); //creo una "maschera" delle dimensioni del mio frame 
    cv::Subdiv2D subdiv(rect);
    /*The function subdiv creates an empty Delaunay subdivision where 2D points can be added using the function insert() .
     All of the points to be added must be within the specified rectangle, otherwise a runtime error is raised.
    */

    // Inserimento punti in subdiv 
    for (const auto &point : good_new) //itero su tutto good_new
    {
        if (rect.contains(point))
        {
            try
            {
                subdiv.insert(point);
            }
            catch (const cv::Exception &e)
            {
                std::cerr << "Error inserting point in Delaunay: " << point << ", " << e.what() << std::endl;
            }
        }
    }

    // Ottieni triangoli e disegna
    std::vector<cv::Vec6f> triangleList;
    try
    {
        subdiv.getTriangleList(triangleList);
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "Error retrieving triangle list: " << e.what() << std::endl;
        return;
    }

    std::vector<cv::Point> pt(3);
    for (size_t i = 0; i < triangleList.size(); i++)
    {
        cv::Vec6f t = triangleList[i];
        pt[0] = cv::Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = cv::Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = cv::Point(cvRound(t[4]), cvRound(t[5]));

        // Disegna triangoli completamente dentro l'immagine
        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            cv::line(current, pt[0], pt[1], color, 1, cv::LINE_AA, 0);
            cv::line(current, pt[1], pt[2], color, 1, cv::LINE_AA, 0);
            cv::line(current, pt[2], pt[0], color, 1, cv::LINE_AA, 0);
        }
    }
}

void Visualizer::drawVoronoi(const cv::Mat &current,
                             std::vector<cv::Point2f> &good_new,
                             const cv::Scalar& color)
{
    int width = current.cols;
    int height = current.rows;

    // --- Rettangolo Subdiv2D leggermente più piccolo per evitare crash ai bordi ---
    cv::Rect safeRect(0, 0, width - 2, height - 2);
    cv::Subdiv2D subdiv(safeRect);

    // --- Filtra e clampa i punti reali ---
    std::vector<cv::Point2f> safe_points;
    for (const auto &p : good_new)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y))
            continue;

        float x = std::min(std::max(p.x, 0.0f), float(width - 3));
        float y = std::min(std::max(p.y, 0.0f), float(height - 3));
        safe_points.push_back(cv::Point2f(x, y));
    }

    // --- Aggiungi punti di cornice per chiudere Voronoi ---
    std::vector<cv::Point2f> borderPts = {
        {0.0f, 0.0f},
        {float(width-3), 0.0f},
        {0.0f, float(height-3)},
        {float(width-3), float(height-3)},
        {float((width-3)/2), 0.0f},
        {float((width-3)/2), float(height-3)},
        {0.0f, float((height-3)/2)},
        {float(width-3), float((height-3)/2)}
    };
    safe_points.insert(safe_points.end(), borderPts.begin(), borderPts.end());

    // --- Controlla almeno 2 punti validi ---
    if (safe_points.size() < 2)
        return;

    // --- Inserisci tutti i punti in Subdiv2D ---
    for (const auto &p : safe_points)
        subdiv.insert(p);

    // --- Ottieni faccette di Voronoi ---
    std::vector<std::vector<cv::Point2f>> facets;
    std::vector<cv::Point2f> centers;
    subdiv.getVoronoiFacetList(std::vector<int>(), facets, centers);

    // --- Disegna faccette e centri ---
    std::vector<cv::Point> ifacet;
    std::vector<std::vector<cv::Point>> ifacets(1);
    for (size_t i = 0; i < facets.size(); ++i)
    {
        ifacet.clear();
        for (size_t j = 0; j < facets[i].size(); ++j)
            ifacet.push_back(cv::Point(cvRound(facets[i][j].x), cvRound(facets[i][j].y)));

        ifacets[0] = ifacet;
        cv::polylines(current, ifacets, true, color, 1, cv::LINE_AA);
        cv::circle(current, centers[i], 2, color, cv::FILLED, cv::LINE_AA);
    }

    // --- opzionale: disegna contorno immagine ---
    cv::rectangle(current, safeRect, cv::Scalar(0,255,0), 1);
}

