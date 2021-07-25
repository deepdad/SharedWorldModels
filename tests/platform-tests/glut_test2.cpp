#include <GL/glut.h>
#include <iostream>
#include<string>
using namespace std;

void displayMe(void)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POLYGON);
        glVertex3f(0.5, 0.0, 0.5);
        glVertex3f(0.5, 0.0, 0.0);
        glVertex3f(0.0, 0.5, 0.0);
        glVertex3f(0.0, 0.0, 0.5);
    glEnd();
    glFlush();
}

int main(int argc, char** argv)
{
    GLint major, minor;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    std::cout << "TEST: " << glGetString(GL_VERSION) <<  "\n";
    std::cout << glGetString(GL_VERSION) << ", " << major << ", " << minor << "\n";
    std::cout << glGetString(GL_VERSION) << ", " << major << ", " << minor << "\n";
    std::cout << glGetString(GL_VERSION) << ", " << major << ", " << minor << "\n";
    /*glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE);
    glutInitWindowSize(400, 300);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Hello world!");
    glutDisplayFunc(displayMe);
    glutMainLoop();*/


    return 0;
}
