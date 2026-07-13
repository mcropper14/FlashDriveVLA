/****************************************************************************
** Meta object code from reading C++ file 'settings.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "settings.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'settings.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_SettingsWindowSP_t {
    QByteArrayData data[1];
    char stringdata0[17];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_SettingsWindowSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_SettingsWindowSP_t qt_meta_stringdata_SettingsWindowSP = {
    {
QT_MOC_LITERAL(0, 0, 16) // "SettingsWindowSP"

    },
    "SettingsWindowSP"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_SettingsWindowSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void SettingsWindowSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject SettingsWindowSP::staticMetaObject = { {
    &SettingsWindow::staticMetaObject,
    qt_meta_stringdata_SettingsWindowSP.data,
    qt_meta_data_SettingsWindowSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *SettingsWindowSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *SettingsWindowSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_SettingsWindowSP.stringdata0))
        return static_cast<void*>(this);
    return SettingsWindow::qt_metacast(_clname);
}

int SettingsWindowSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = SettingsWindow::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_TogglesPanelSP_t {
    QByteArrayData data[5];
    char stringdata0[40];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_TogglesPanelSP_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_TogglesPanelSP_t qt_meta_stringdata_TogglesPanelSP = {
    {
QT_MOC_LITERAL(0, 0, 14), // "TogglesPanelSP"
QT_MOC_LITERAL(1, 15, 11), // "updateState"
QT_MOC_LITERAL(2, 27, 0), // ""
QT_MOC_LITERAL(3, 28, 9), // "UIStateSP"
QT_MOC_LITERAL(4, 38, 1) // "s"

    },
    "TogglesPanelSP\0updateState\0\0UIStateSP\0"
    "s"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_TogglesPanelSP[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   19,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 3,    4,

       0        // eod
};

void TogglesPanelSP::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<TogglesPanelSP *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->updateState((*reinterpret_cast< const UIStateSP(*)>(_a[1]))); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject TogglesPanelSP::staticMetaObject = { {
    &TogglesPanel::staticMetaObject,
    qt_meta_stringdata_TogglesPanelSP.data,
    qt_meta_data_TogglesPanelSP,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *TogglesPanelSP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *TogglesPanelSP::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_TogglesPanelSP.stringdata0))
        return static_cast<void*>(this);
    return TogglesPanel::qt_metacast(_clname);
}

int TogglesPanelSP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = TogglesPanel::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
